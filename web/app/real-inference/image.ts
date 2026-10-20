import { pillowBilinearGray } from "./pillow-resize";

export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const MAX_SOURCE_SIDE = 8192;
export const MAX_SOURCE_PIXELS = 4_000_000;

export type ProcessedImage = { sourceGray: Uint8Array; sourceWidth: number; sourceHeight: number; pixels: Uint8Array };
export type DecodedRgbImage = { rgb: Uint8Array; width: number; height: number };

/** Decode JPEG, PNG, or WebP into the browser's sRGB uint8 RGB pixels. */
export async function imageToRgb(file: File): Promise<DecodedRgbImage> {
  if (!new Set(["image/jpeg", "image/png", "image/webp"]).has(file.type.toLowerCase())) throw new Error("Choose a JPEG, PNG, or WebP image.");
  if (!file.size || file.size > MAX_IMAGE_BYTES) throw new Error("Choose a non-empty image smaller than 20 MiB.");
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "none", colorSpaceConversion: "none", premultiplyAlpha: "none" });
  } catch {
    throw new Error("The selected image could not be decoded by this browser.");
  }
  try {
    if (bitmap.width < 1 || bitmap.height < 1 || Math.max(bitmap.width, bitmap.height) > MAX_SOURCE_SIDE || bitmap.width * bitmap.height > MAX_SOURCE_PIXELS) {
      throw new Error("Image dimensions must be at most 8192 pixels per side and 4 megapixels total.");
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    if (!context) throw new Error("Canvas image processing is unavailable in this browser.");
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const rgb = new Uint8Array(bitmap.width * bitmap.height * 3);
    for (let i = 0, p = 0; i < rgb.length; i += 3, p += 4) {
      rgb[i] = rgba[p]; rgb[i + 1] = rgba[p + 1]; rgb[i + 2] = rgba[p + 2];
    }
    return { rgb, width: bitmap.width, height: bitmap.height };
  } finally { bitmap.close(); }
}

/** Decode, convert source pixels to Pillow-style grayscale, and apply Pillow bilinear resize. */
export async function imageToGrayscale(file: File): Promise<ProcessedImage> {
  if (!file.size || file.size > MAX_IMAGE_BYTES) throw new Error("Choose a non-empty image smaller than 20 MiB.");
  const bitmap = await createImageBitmap(file, {
    imageOrientation: "none",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  try {
    if (bitmap.width < 1 || bitmap.height < 1 || Math.max(bitmap.width, bitmap.height) > MAX_SOURCE_SIDE || bitmap.width * bitmap.height > MAX_SOURCE_PIXELS) throw new Error("Image dimensions must be at most 8192 pixels per side and 4 megapixels total.");
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    if (!context) throw new Error("Canvas image processing is unavailable in this browser.");
    context.drawImage(bitmap, 0, 0);
    const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const gray = new Uint8Array(bitmap.width * bitmap.height);
    for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
      // Integer coefficients used by Pillow's convert('L') for RGB input.
      gray[i] = (19595 * rgba[p] + 38470 * rgba[p + 1] + 7471 * rgba[p + 2] + 32768) >>> 16;
    }
    return { sourceGray: gray, sourceWidth: bitmap.width, sourceHeight: bitmap.height, pixels: pillowBilinearGray(gray, bitmap.width, bitmap.height) };
  } finally { bitmap.close(); }
}
