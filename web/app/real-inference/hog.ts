/** scikit-image 0.26 compatible HOG for 128x128 uint8 grayscale input. */
export const IMAGE_SIDE = 128;
export const HOG_SIZE = 8100;

export function extractHog(pixels: ArrayLike<number>): Float32Array {
  if (pixels.length !== IMAGE_SIDE * IMAGE_SIDE) throw new Error("Expected 128 x 128 grayscale pixels.");
  for (let i = 0; i < pixels.length; i++) {
    if (!Number.isFinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 255) throw new Error("Grayscale pixels must be finite values from 0 to 255.");
  }
  const hist = new Float64Array(16 * 16 * 9);
  const at = (r: number, c: number, o: number) => (r * 16 + c) * 9 + o;
  for (let r = 0; r < 128; r++) for (let c = 0; c < 128; c++) {
    const gx = c === 0 || c === 127 ? 0 : pixels[r * 128 + c + 1] - pixels[r * 128 + c - 1];
    const gy = r === 0 || r === 127 ? 0 : pixels[(r + 1) * 128 + c] - pixels[(r - 1) * 128 + c];
    if (gx === 0 && gy === 0) continue;
    let angle = Math.atan2(gy, gx) % Math.PI;
    if (angle < 0) angle += Math.PI;
    const bin = Math.min(Math.floor(angle * (9 / Math.PI)), 8);
    hist[at(Math.floor(r / 8), Math.floor(c / 8), bin)] += Math.hypot(gx, gy);
  }
  const output = new Float32Array(HOG_SIZE);
  let cursor = 0;
  // skimage block iteration/ravel order: block row, block col, cell row, cell col, orientation.
  for (let br = 0; br < 15; br++) for (let bc = 0; bc < 15; bc++) {
    const block = new Float64Array(36);
    let k = 0;
    for (let cr = 0; cr < 2; cr++) for (let cc = 0; cc < 2; cc++) for (let o = 0; o < 9; o++) block[k++] = hist[at(br + cr, bc + cc, o)];
    let norm = Math.sqrt(block.reduce((sum, v) => sum + v * v, 0) + 1e-10);
    for (let i = 0; i < block.length; i++) block[i] = Math.min(block[i] / norm, 0.2);
    norm = Math.sqrt(block.reduce((sum, v) => sum + v * v, 0) + 1e-10);
    for (const value of block) output[cursor++] = value / norm;
  }
  return output;
}
