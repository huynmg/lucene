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
package org.apache.lucene.util.quantization;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;
import static org.apache.lucene.util.quantization.OptimizedScalarQuantizer.MINIMUM_MSE_GRID;

import java.util.Arrays;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;

public class TestOptimizedScalarQuantizer extends LuceneTestCase {

  public void testTransposeHalfByte() {

    byte[] input = new byte[] {8,15,10,7,4,0,9,9};
    
    byte[] output = new byte[4]; // 4 sections for 4 bits

    OptimizedScalarQuantizer.transposeHalfByte(input, output);

    // Expected output calculation:
    // For bit 0 (LSB):     [0,1,0,1,0,0,1,1] = 0b01010011 = 83
    // For bit 1 (2nd bit): [0,1,1,1,0,0,0,0] = 0b01110000 = 112
    // For bit 2 (3rd bit): [0,1,0,1,1,0,0,0] = 0b01011000 = 88 
    // For bit 3 (MSB):     [1,1,1,0,0,0,1,1] = 0b11100011 = -29
    byte[] expected = new byte[] {83, 112, 88, -29};
    assertArrayEquals(expected, output);
  }

  public void testTransposeByte() {
    byte[] input = new byte[] {8,15,10,7,4,0,9,9};
    byte[] output = new byte[8]; // 4 sections for 4 bits


    // Expected output calculation:
    // For bit 0 :     [0,1,0,1,0,0,1,1] = 0b01010011 = 83
    // For bit 1 :     [0,1,1,1,0,0,0,0] = 0b01110000 = 112
    // For bit 2 :     [0,1,0,1,1,0,0,0] = 0b01011000 = 88 
    // For bit 3 :     [1,1,1,0,0,0,1,1] = 0b11100011 = -29
    // For bit 4 :     [0,0,0,0,0,0,0,0] = 0b00000000 = 0
    // For bit 5 :     [0,0,0,0,0,0,0,0] = 0b01110000 = 0
    // For bit 6 :     [0,0,0,0,0,0,0,0] = 0b01011000 = 0 
    // For bit 7 :     [0,0,0,0,0,0,0,0] = 0b11100011 = 0
    OptimizedScalarQuantizer.transposeByte(input, output);
    byte[] expected = new byte[] {83, 112, 88, -29, 0, 0, 0, 0};
    assertArrayEquals(expected, output);
  }
}
    