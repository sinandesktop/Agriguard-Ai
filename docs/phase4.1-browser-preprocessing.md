# Phase 4.1 browser preprocessing audit

## Fixture audit

The audited directory contains 12 entries in `fixtures.json` and 12 `.jpg` files in `images/`: two each for Potato early blight, Potato late blight, Potato healthy, Tomato early blight, Tomato late blight, and Tomato healthy. Each entry's manifest path exists under `ml/data/plantvillage-source/raw/color`; each copied fixture's SHA-256 matched those exact source bytes and its held-out `test.csv` row label. The directory was complete, so no fixture regeneration was needed to repair a missing-file problem. A file-picker view showing only 6-8 entries does not establish that the directory is incomplete; likely the picker was scrolled, filtered, or opened at a different directory, but the precise cause on the user's machine cannot be determined here.

The earlier browser harness matched references by filename suffix. That was ambiguous: a different image named `00.jpg` could have been compared to a fixture. Fixture JSON now stores the original source SHA-256 and source dimensions; the browser computes the selected file SHA-256 and only compares it to an exact hash match. Non-fixture images explicitly report that no reference comparison was made.

## Contract versus browser implementation

Python `preprocess_image` opens the file in Pillow, ignores EXIF orientation, converts to `L`, then applies Pillow's 8-bit bilinear resize to 128 x 128. RGB-to-L uses Pillow's fixed-point ITU-R 601 coefficients `19595, 38470, 7471` with a `32768` rounding offset. The browser decoder produces RGBA channel bytes; browser preprocessing reads R, G, B in that order, uses the same integer formula/rounding on the original-resolution pixels, and ignores alpha as Pillow's RGB-to-L conversion does. Browser image decoding and color management can still differ from Pillow for particular formats or profiles; the new source grayscale comparison exposes that difference.

Canvas resizing had been used previously. It combines browser-dependent interpolation with grayscale conversion after resize, while Python converts to grayscale before resizing. The new browser code does not resize with Canvas. It uses a TypeScript port of Pillow's bilinear resampler: center-aligned sampling, widening the triangular support when shrinking (antialiasing), normalized coefficients quantized to 22-bit fixed point, integer accumulation and rounding after each axis pass, and Pillow's axis-pass selection. It preserves the original trained preprocessing contract rather than changing model training.

HOG is unchanged: central differences with zero outer borders; unsigned angle modulo 180 degrees; hard assignment to 9 bins; 8 x 8 cell histograms; blocks traversed row-major with cell-row/cell-column/bin order; L2-Hys with `sqrt(sum(square) + 1e-10)`, clipping at 0.2 and second normalization; C-order flattened Float32 output. Python and TypeScript HOG compare on the same 128 x 128 grayscale fixtures.

## Verification

`npm run test:hog` compares Python's resized grayscale arrays with the TypeScript Pillow resizer and compares Python's HOG arrays with TypeScript HOG. These tests isolate resize and HOG using Python-generated source grayscale values. They do not test browser image decoding. The page reports source grayscale differences, resized pixel differences, HOG differences, decision-score differences, and browser/Python saved predictions for an exact-hash matched fixture.

Relative errors divide by `max(abs(reference), floor)`. The display uses floor 1 for uint8 pixels and `1e-6` for normalized HOG/decision values. Relative error can be extremely large when a reference feature is near zero; assess it alongside absolute max/mean errors. No arbitrary end-to-end browser tolerance is treated as a parity pass.

## Pillow source reference

The fixed-point resampling details were checked against Pillow's `Resample.c` and RGB-to-L conversion in `Convert.c`: [resampler implementation](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c), [channel conversion implementation](https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Convert.c). The model's actual behavior remains defined by its recorded Python Pillow version and `preprocessing.py`.
