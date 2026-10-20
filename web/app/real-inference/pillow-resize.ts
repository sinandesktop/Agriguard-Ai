const TARGET = 128;
const PRECISION_BITS = 22;
const FIXED_ONE = 1 << PRECISION_BITS;
const HALF_FIXED = 1 << (PRECISION_BITS - 1);

type AxisWeights = { starts: Int32Array; lengths: Int32Array; values: Int32Array; width: number };

function axisWeights(inputSize: number, outputSize: number): AxisWeights {
  const scale = inputSize / outputSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale;
  const width = Math.ceil(support) * 2 + 1;
  const starts = new Int32Array(outputSize);
  const lengths = new Int32Array(outputSize);
  const values = new Int32Array(outputSize * width);
  for (let out = 0; out < outputSize; out++) {
    const center = (out + 0.5) * scale;
    let min = Math.trunc(center - support + 0.5);
    let max = Math.trunc(center + support + 0.5);
    min = Math.max(min, 0);
    max = Math.min(max, inputSize);
    const count = max - min;
    starts[out] = min;
    lengths[out] = count;
    const row = out * width;
    const raw = new Float64Array(count);
    let total = 0;
    for (let tap = 0; tap < count; tap++) {
      const distance = Math.abs((tap + min - center + 0.5) / filterScale);
      const weight = distance < 1 ? 1 - distance : 0;
      raw[tap] = weight;
      total += weight;
    }
    if (total !== 0) {
      for (let tap = 0; tap < count; tap++) {
        values[row + tap] = Math.floor((raw[tap] / total) * FIXED_ONE + 0.5);
      }
    }
  }
  return { starts, lengths, values, width };
}

function horizontal(src: Uint8Array, width: number, height: number): Uint8Array {
  const weights = axisWeights(width, TARGET);
  const dst = new Uint8Array(TARGET * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < TARGET; x++) {
    const taps = x * weights.width;
    let sum = HALF_FIXED;
    for (let k = 0; k < weights.lengths[x]; k++) sum += src[y * width + weights.starts[x] + k] * weights.values[taps + k];
    dst[y * TARGET + x] = Math.max(0, Math.min(255, sum >> PRECISION_BITS));
  }
  return dst;
}

function vertical(src: Uint8Array, width: number, height: number): Uint8Array {
  const weights = axisWeights(height, TARGET);
  const dst = new Uint8Array(width * TARGET);
  for (let y = 0; y < TARGET; y++) for (let x = 0; x < width; x++) {
    const taps = y * weights.width;
    let sum = HALF_FIXED;
    for (let k = 0; k < weights.lengths[y]; k++) sum += src[(weights.starts[y] + k) * width + x] * weights.values[taps + k];
    dst[y * width + x] = Math.max(0, Math.min(255, sum >> PRECISION_BITS));
  }
  return dst;
}

/** Pillow's 8-bit bilinear resize: antialiasing on shrink, fixed-point weights and rounding. */
export function pillowBilinearGray(source: Uint8Array, width: number, height: number): Uint8Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || source.length !== width * height) throw new Error("Invalid grayscale source dimensions.");
  if (width === TARGET && height === TARGET) return source.slice();
  // Pillow picks the vertical pass first only when its shrink is more than twice the horizontal shrink.
  const verticalFirst = height - TARGET > 0 && height - TARGET > (width - TARGET) * 2;
  if (!verticalFirst) return vertical(horizontal(source, width, height), TARGET, height);
  return horizontal(vertical(source, width, height), width, TARGET);
}

/** Pillow-equivalent per-channel resize for interleaved RGB uint8 pixels. */
export function pillowBilinearRgb(source: Uint8Array, width: number, height: number): Uint8Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || source.length !== width * height * 3) {
    throw new Error("Invalid RGB source dimensions.");
  }
  const channels = [new Uint8Array(width * height), new Uint8Array(width * height), new Uint8Array(width * height)];
  for (let i = 0, j = 0; i < source.length; i += 3, j++) {
    channels[0][j] = source[i];
    channels[1][j] = source[i + 1];
    channels[2][j] = source[i + 2];
  }
  const resized = channels.map((channel) => pillowBilinearGray(channel, width, height));
  const output = new Uint8Array(TARGET * TARGET * 3);
  for (let i = 0, j = 0; i < output.length; i += 3, j++) {
    output[i] = resized[0][j];
    output[i + 1] = resized[1][j];
    output[i + 2] = resized[2][j];
  }
  return output;
}
