#pragma once
#include <stddef.h>   
#include <stdint.h>  

#ifdef __cplusplus
extern "C" {
#endif

// Mirror the compression methods provided by TerseTS.
enum Method {
  PoorMansCompressionMidrange   = 0,
  PoorMansCompressionMean       = 1,
  SwingFilter                   = 2,
  SwingFilterDisconnected       = 3,
  SlideFilter                   = 4,
  SimPiece                      = 5,
  PiecewiseConstantHistogram    = 6,
  PiecewiseLinearHistogram      = 7,
  ABCLinearApproximation        = 8,
  VisvalingamWhyatt             = 9,
  SlidingWindow                 = 10,
  BottomUp                      = 11,
  MixPiece                      = 12,
  BitPackedQuantization         = 13,
  RunLengthEncoding             = 14,
  NonLinearApproximation        = 15,
  SerfQT                        = 16,
  DiscreteFourierTransform      = 17,
};

// Read-only view of input data (the library will not modify it).
struct UncompressedValues {
  const double *data;
  size_t        len;
};

// Output buffer for compressed bytes (the library writes these fields).
struct CompressedValues {
  uint8_t *data;   
  size_t   len;    
};

// Compress uncompressed_values to compressed_values according to configuration.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t compress(struct UncompressedValues uncompressed_values,
                 struct CompressedValues *compressed_values,
                 uint8_t method, 
                 const char *configuration);

// Decompress compressed_values to uncompressed_values according to configuration.
// Returns 0 on success, non-zero on error (e.g., 1 = unsupported method).
int32_t decompress(struct CompressedValues compressed_values,
                   struct UncompressedValues *uncompressed_values);

// Free functions.
void freeCompressedValues(struct CompressedValues *compressed_values);
void freeUncompressedValues(struct UncompressedValues *uncompressed_values);

#ifdef __cplusplus
}
#endif
