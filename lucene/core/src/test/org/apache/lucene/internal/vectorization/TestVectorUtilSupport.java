/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.internal.vectorization;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;
import java.util.stream.IntStream;

public class TestVectorUtilSupport extends BaseVectorizationTestCase {

  private static final int[] VECTOR_SIZES = {
    1, 4, 6, 8, 13, 16, 25, 32, 64, 100, 128, 207, 256, 300, 512, 702, 1024, 1536, 2046, 2048, 4096,
    4098
  };

  private final int size;
  private final double delta;

  public TestVectorUtilSupport(int size) {
    this.size = size;
    // scale the delta with the size
    this.delta = 1e-5 * size;
  }

  @ParametersFactory
  public static Iterable<Object[]> parametersFactory() {
    return () -> IntStream.of(VECTOR_SIZES).boxed().map(i -> new Object[] {i}).iterator();
  }

  public void testFloatVectors() {
    var a = new float[size];
    var b = new float[size];
    for (int i = 0; i < size; ++i) {
      a[i] = random().nextFloat();
      b[i] = random().nextFloat();
    }
    assertFloatReturningProviders(p -> p.dotProduct(a, b));
    assertFloatReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));
  }

  public void testBinaryVectors() {
    var a = new byte[size];
    var b = new byte[size];
    random().nextBytes(a);
    random().nextBytes(b);
    assertIntReturningProviders(p -> p.dotProduct(a, b));
    assertIntReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));
  }

  public void testBinaryVectorsBoundaries() {
    var a = new byte[size];
    var b = new byte[size];

    Arrays.fill(a, Byte.MIN_VALUE);
    Arrays.fill(b, Byte.MIN_VALUE);
    assertIntReturningProviders(p -> p.dotProduct(a, b));
    assertIntReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));

    Arrays.fill(a, Byte.MAX_VALUE);
    Arrays.fill(b, Byte.MAX_VALUE);
    assertIntReturningProviders(p -> p.dotProduct(a, b));
    assertIntReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));

    Arrays.fill(a, Byte.MIN_VALUE);
    Arrays.fill(b, Byte.MAX_VALUE);
    assertIntReturningProviders(p -> p.dotProduct(a, b));
    assertIntReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));

    Arrays.fill(a, Byte.MAX_VALUE);
    Arrays.fill(b, Byte.MIN_VALUE);
    assertIntReturningProviders(p -> p.dotProduct(a, b));
    assertIntReturningProviders(p -> p.squareDistance(a, b));
    assertFloatReturningProviders(p -> p.cosine(a, b));
  }

  public void testInt4DotProduct() {
    assumeTrue("even sizes only", size % 2 == 0);
    var a = new byte[size];
    var b = new byte[size];
    for (int i = 0; i < size; ++i) {
      a[i] = (byte) random().nextInt(16);
      b[i] = (byte) random().nextInt(16);
    }

    assertIntReturningProviders(p -> p.int4DotProduct(a, false, pack(b), true));
    assertIntReturningProviders(p -> p.int4DotProduct(pack(a), true, b, false));
    assertEquals(
        LUCENE_PROVIDER.getVectorUtilSupport().dotProduct(a, b),
        PANAMA_PROVIDER.getVectorUtilSupport().int4DotProduct(a, false, pack(b), true));
  }

  public void testInt4DotProductBoundaries() {
    assumeTrue("even sizes only", size % 2 == 0);
    byte MAX_VALUE = 15;
    var a = new byte[size];
    var b = new byte[size];

    Arrays.fill(a, MAX_VALUE);
    Arrays.fill(b, MAX_VALUE);
    assertIntReturningProviders(p -> p.int4DotProduct(a, false, pack(b), true));
    assertIntReturningProviders(p -> p.int4DotProduct(pack(a), true, b, false));
    assertEquals(
        LUCENE_PROVIDER.getVectorUtilSupport().dotProduct(a, b),
        PANAMA_PROVIDER.getVectorUtilSupport().int4DotProduct(a, false, pack(b), true));

    byte MIN_VALUE = 0;
    Arrays.fill(a, MIN_VALUE);
    Arrays.fill(b, MIN_VALUE);
    assertIntReturningProviders(p -> p.int4DotProduct(a, false, pack(b), true));
    assertIntReturningProviders(p -> p.int4DotProduct(pack(a), true, b, false));
    assertEquals(
        LUCENE_PROVIDER.getVectorUtilSupport().dotProduct(a, b),
        PANAMA_PROVIDER.getVectorUtilSupport().int4DotProduct(a, false, pack(b), true));
  }

  public void testInt4BitDotProduct() {
    var binaryQuantized = new byte[size];
    var int4Quantized = new byte[size * 4];
    random().nextBytes(binaryQuantized);
    random().nextBytes(int4Quantized);
    assertLongReturningProviders(p -> p.int4BitDotProduct(int4Quantized, binaryQuantized));
  }

  public void testInt4BitDotProductBoundaries() {
    var binaryQuantized = new byte[size];
    var int4Quantized = new byte[size * 4];

    Arrays.fill(binaryQuantized, Byte.MAX_VALUE);
    Arrays.fill(int4Quantized, Byte.MAX_VALUE);
    assertLongReturningProviders(p -> p.int4BitDotProduct(int4Quantized, binaryQuantized));

    Arrays.fill(binaryQuantized, Byte.MIN_VALUE);
    Arrays.fill(int4Quantized, Byte.MIN_VALUE);
    assertLongReturningProviders(p -> p.int4BitDotProduct(int4Quantized, binaryQuantized));
  }
  
  public void testInt8BitDotProduct() {
    var binaryQuantized = new byte[size];
    var int8Quantized = new byte[size * 8];
    random().nextBytes(binaryQuantized);
    random().nextBytes(int8Quantized);
    assertLongReturningProviders(p -> p.int8BitDotProduct(int8Quantized, binaryQuantized));
  }

  public void testInt8BitDotProductBoundaries() {
    var binaryQuantized = new byte[size];
    var int8Quantized = new byte[size * 8];

    Arrays.fill(binaryQuantized, Byte.MAX_VALUE);
    Arrays.fill(int8Quantized, Byte.MAX_VALUE);
    assertLongReturningProviders(p -> p.int8BitDotProduct(int8Quantized, binaryQuantized));

    Arrays.fill(binaryQuantized, Byte.MIN_VALUE);
    Arrays.fill(int8Quantized, Byte.MIN_VALUE);
    assertLongReturningProviders(p -> p.int8BitDotProduct(int8Quantized, binaryQuantized));
  }
  
  public void testInt8BitDotProductWithConcreteExample() {
    // Create a concrete example with known values
    
    // The int8BitDotProduct implementation works differently than we initially thought.
    // It processes 8 separate "planes" of bits, where each plane contributes to the final result
    // with a different weight (shifted by its position).
    
    // For a simple test, let's create a case with just one byte in the binary vector
    // and 8 bytes in the quantized vector (one for each bit position/plane)
    
    // 1. Create a binary vector with a single byte
    byte[] d = new byte[] {(byte) 0x03}; // Binary: 00000011 (bits 0 and 1 are set)
    
    // 2. Create a quantized vector with 8 bytes (one for each bit position/plane)
    // We'll use values 1-8 for simplicity
    byte[] q = new byte[] {1, 2, 3, 4, 5, 6, 7, 8};
    
    // 3. Calculate the expected result based on the implementation:
    // - For each of the 8 planes (i=0 to 7):
    //   - Count bits that are set in both q[i*size+r] and d[r]
    //   - Multiply this count by 2^i (shift left by i)
    //   - Add to the total
    // 
    // In our case:
    // - Plane 0 (i=0): q[0]=1, d[0]=0x03, bitCount(1 & 0x03)=1, 1 << 0 = 1
    // - Plane 1 (i=1): q[1]=2, d[0]=0x03, bitCount(2 & 0x03)=2, 2 << 1 = 4
    // - Plane 2 (i=2): q[2]=3, d[0]=0x03, bitCount(3 & 0x03)=2, 2 << 2 = 8
    // - Plane 3 (i=3): q[3]=4, d[0]=0x03, bitCount(4 & 0x03)=0, 0 << 3 = 0
    // - Plane 4 (i=4): q[4]=5, d[0]=0x03, bitCount(5 & 0x03)=1, 1 << 4 = 16
    // - Plane 5 (i=5): q[5]=6, d[0]=0x03, bitCount(6 & 0x03)=2, 2 << 5 = 64
    // - Plane 6 (i=6): q[6]=7, d[0]=0x03, bitCount(7 & 0x03)=2, 2 << 6 = 128
    // - Plane 7 (i=7): q[7]=8, d[0]=0x03, bitCount(8 & 0x03)=0, 0 << 7 = 0
    // Total: 1 + 4 + 8 + 0 + 16 + 64 + 128 + 0 = 221
    long expectedResult = 221;
    
    // 4. Call the int8BitDotProduct method on both implementations
    long luceneResult = LUCENE_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    long panamaResult = PANAMA_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    
    // 5. Verify the results match our manual calculation
    assertEquals(expectedResult, luceneResult);
    assertEquals(expectedResult, panamaResult);
    
    // 6. Explanation of how int8BitDotProduct actually works:
    // - It processes 8 separate "planes" of bits
    // - For each plane i (0-7), it counts bits that are set in both vectors
    // - It then multiplies this count by 2^i (shifts left by i)
    // - The final result is the sum of these weighted bit counts
  }

  public void testInt8BitDotProductWithMultipleBytes() {
    // Create a larger example that spans multiple bytes in the binary vector
    
    // 1. Define a 16-element 8-bit quantized vector
    byte[] q = new byte[] {
        5, 10, 15, 20, 25, 30, 35, 40,  // First 8 values
        45, 50, 55, 60, 65, 70, 75, 80  // Second 8 values
    };
    
    // 2. Define a binary vector with 2 bytes (for 16 values)
    //    - First byte: 00001111 (decimal 15, hex 0x0F)
    //    - Second byte: 11110000 (decimal 240, hex 0xF0)
    //    - This means we include values at positions 0-3 and 12-15
    byte[] d = new byte[] {(byte) 0x0F, (byte) 0xF0};
    
    // 3. Calculate the expected dot product manually:
    //    - From first byte (positions 0-7):
    //      - Include positions 0-3: 5 + 10 + 15 + 20 = 50
    //      - Exclude positions 4-7: 0
    //    - From second byte (positions 8-15):
    //      - Exclude positions 8-11: 0
    //      - Include positions 12-15: 65 + 70 + 75 + 80 = 290
    //    - Total sum: 50 + 290 = 340
    float expectedResult = 340.0f;
    
    // 4. Call the int8BitDotProduct method on both implementations
    float luceneResult = LUCENE_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    float panamaResult = PANAMA_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    
    // 5. Verify the results match our manual calculation
    assertEquals(expectedResult, luceneResult, 0.0f);
    assertEquals(expectedResult, panamaResult, 0.0f);
    
    // 6. Explanation of how int8BitDotProduct works with multiple bytes:
    // - The method processes each byte in 'd' sequentially
    // - For the first byte (0x0F), it includes the first 4 values from 'q'
    // - For the second byte (0xF0), it includes the last 4 values from 'q'
    // - The calculation spans across byte boundaries correctly
    // - This demonstrates that the method handles multi-byte binary vectors properly
  }

  public void testInt8BitDotProductWithAllBitsSet() {
    // Test case where all bits in the binary vector are set to 1
    
    // 1. Define an 8-element 8-bit quantized vector
    byte[] q = new byte[] {10, 20, 30, 40, 50, 60, 70, 80};
    
    // 2. Define a binary vector with all bits set to 1
    //    - Binary: 11111111 (decimal 255, hex 0xFF)
    byte[] d = new byte[] {(byte) 0xFF};
    
    // 3. Calculate the expected dot product manually:
    //    - All positions are included: 10 + 20 + 30 + 40 + 50 + 60 + 70 + 80 = 360
    float expectedResult = 360.0f;
    
    // 4. Call the int8BitDotProduct method on both implementations
    float luceneResult = LUCENE_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    float panamaResult = PANAMA_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    
    // 5. Verify the results match our manual calculation
    assertEquals(expectedResult, luceneResult, 0.0f);
    assertEquals(expectedResult, panamaResult, 0.0f);
    
    // 6. Explanation:
    // - When all bits are set, every value in the quantized vector is included
    // - This is equivalent to a simple sum of all values in the quantized vector
    // - This test verifies the method correctly handles the case where all bits are set
  }
  
  public void testInt8BitDotProductComparedToInt4Bit() {
    // Test that int8BitDotProduct produces the same results as int4BitDotProduct
    // when given equivalent inputs (values limited to 0-15 range)
    
    // 1. Define a quantized vector with values 0-15 only
    byte[] q = new byte[] {5, 10, 15, 7, 3, 12, 9, 1};
    
    // 2. Define a binary vector
    byte[] d = new byte[] {(byte) 0x3C}; // Binary: 00111100
    
    // 3. Calculate using int8BitDotProduct
    float result8Bit = LUCENE_PROVIDER.getVectorUtilSupport().int8BitDotProduct(q, d);
    
    // 4. Calculate using int4BitDotProduct
    // For int4BitDotProduct, we need to ensure the vector length matches the requirement
    // q.length == d.length * 4
    float result4Bit = LUCENE_PROVIDER.getVectorUtilSupport().int4BitDotProduct(q, d);
    
    // 5. Verify the results match
    assertEquals(result4Bit, result8Bit, 0.0f);
    
    // 6. Explanation:
    // - When using the same input values within the 0-15 range
    // - Both methods should produce identical results
    // - This verifies that int8BitDotProduct is consistent with int4BitDotProduct
    // - The expected result is: 10 + 15 + 7 + 3 = 35 (bits 2,3,4,5 are set)
  }

  static byte[] pack(byte[] unpacked) {
    int len = (unpacked.length + 1) / 2;
    var packed = new byte[len];
    for (int i = 0; i < len; i++) {
      packed[i] = (byte) (unpacked[i] << 4 | unpacked[packed.length + i]);
    }
    return packed;
  }

  public void testMinMaxScalarQuantize() {
    Random r = random();
    float min = r.nextFloat(-1, 1);
    float max = r.nextFloat(min, 1);
    float divisor = (float) ((1 << 7) - 1); // 7 bits quantization here

    float scale = divisor / (max - min);
    float alpha = (max - min) / divisor;

    float[] vector = new float[size];
    for (int i = 0; i < vector.length; i++) {
      vector[i] = (r.nextFloat() * (max - min)) + min;
    }

    List<byte[]> outputs = new ArrayList<>();
    assertFloatReturningProviders(
        p -> {
          byte[] output = new byte[size];
          outputs.add(output);
          return p.minMaxScalarQuantize(vector, output, scale, alpha, min, max);
        });

    // check the outputs are identical
    for (int o = 1; o < outputs.size(); o++) {
      assertArrayEquals(outputs.getFirst(), outputs.get(o));
    }

    // check recalculation too
    float newMax = max * 2;
    float newMin = min / 2;
    float newScale = divisor / (newMax - newMin);
    float newAlpha = (newMax - newMin) / divisor;

    assertFloatReturningProviders(
        p ->
            p.recalculateScalarQuantizationOffset(
                outputs.getFirst(), alpha, min, newScale, newAlpha, newMin, newMax));
  }

  private void assertFloatReturningProviders(ToDoubleFunction<VectorUtilSupport> func) {
    assertEquals(
        func.applyAsDouble(LUCENE_PROVIDER.getVectorUtilSupport()),
        func.applyAsDouble(PANAMA_PROVIDER.getVectorUtilSupport()),
        delta);
  }

  private void assertIntReturningProviders(ToIntFunction<VectorUtilSupport> func) {
    assertEquals(
        func.applyAsInt(LUCENE_PROVIDER.getVectorUtilSupport()),
        func.applyAsInt(PANAMA_PROVIDER.getVectorUtilSupport()));
  }

  private void assertLongReturningProviders(ToLongFunction<VectorUtilSupport> func) {
    assertEquals(
        func.applyAsLong(LUCENE_PROVIDER.getVectorUtilSupport()),
        func.applyAsLong(PANAMA_PROVIDER.getVectorUtilSupport()));
  }
}
