import { extractHog } from "./hog";
import { pillowBilinearGray, pillowBilinearRgb } from "./pillow-resize";

export const PHASE9_FEATURE_COUNT = 8132;
export const PHASE9_HOG_COUNT = 8100;
export const PHASE9_HSV_COUNT = 32;

export type Phase9Features = {
  hog: Float32Array;
  hsv: Float32Array;
  combined: Float32Array;
};

function assertRgb(rgb: ArrayLike<number>, width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || rgb.length !== width * height * 3) {
    throw new Error("Expected interleaved RGB pixels with valid image dimensions.");
  }
  for (let i = 0; i < rgb.length; i++) {
    const value = rgb[i];
    if (!Number.isInteger(value) || value < 0 || value > 255) throw new Error("RGB channels must be integer values from 0 to 255.");
  }
}

function rgbToPillowGray(rgb: ArrayLike<number>): Uint8Array {
  const gray = new Uint8Array(rgb.length / 3);
  for (let i = 0, p = 0; i < rgb.length; i += 3, p++) {
    // Pillow RGB -> L integer coefficients, preserving its uint8 rounding.
    gray[p] = (19595 * rgb[i] + 38470 * rgb[i + 1] + 7471 * rgb[i + 2] + 32768) >>> 16;
  }
  return gray;
}

function hsvHistogram(rgb: Uint8Array): Float32Array {
  const counts = [new Uint32Array(16), new Uint32Array(8), new Uint32Array(8)];
  for (let i = 0; i < rgb.length; i += 3) {
    // scikit-image receives float32 RGB values in [0, 1]. Keep float32
    // intermediates so values near NumPy histogram boundaries stay aligned.
    const red = Math.fround(rgb[i] / 255);
    const green = Math.fround(rgb[i + 1] / 255);
    const blue = Math.fround(rgb[i + 2] / 255);
    const value = Math.max(red, green, blue);
    const minimum = Math.min(red, green, blue);
    const delta = Math.fround(value - minimum);

    let hue = 0;
    if (delta !== 0) {
      if (red === value) hue = Math.fround(Math.fround(green - blue) / delta);
      if (green === value) hue = Math.fround(2 + Math.fround(Math.fround(blue - red) / delta));
      if (blue === value) hue = Math.fround(4 + Math.fround(Math.fround(red - green) / delta));
      const hueFraction = Math.fround(hue / 6);
      const wrappedHue = hueFraction % 1;
      hue = Math.fround(wrappedHue < 0 ? wrappedHue + 1 : wrappedHue);
    }
    const saturation = value === 0 ? 0 : Math.fround(delta / value);
    const add = (hist: Uint32Array, channel: number) => {
      const bin = Math.min(Math.floor(channel * hist.length), hist.length - 1);
      hist[bin]++;
    };
    add(counts[0], hue);
    add(counts[1], saturation);
    add(counts[2], value);
  }
  const output = new Float32Array(PHASE9_HSV_COUNT);
  const pixelCount = rgb.length / 3;
  let offset = 0;
  for (const channel of counts) {
    for (const count of channel) output[offset++] = count / pixelCount;
  }
  return output;
}

/** Extract the frozen Phase 9 representation from decoded RGB uint8 source pixels. */
export function extractPhase9Features(rgb: ArrayLike<number>, width: number, height: number): Phase9Features {
  assertRgb(rgb, width, height);
  const gray = rgbToPillowGray(rgb);
  const hogPixels = pillowBilinearGray(gray, width, height);
  const hog = extractHog(hogPixels);
  const resizedRgb = pillowBilinearRgb(Uint8Array.from(rgb), width, height);
  const hsv = hsvHistogram(resizedRgb);
  const combined = new Float32Array(PHASE9_FEATURE_COUNT);
  combined.set(hog, 0);
  combined.set(hsv, PHASE9_HOG_COUNT);
  return { hog, hsv, combined };
}
