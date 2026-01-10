# Flat HNSW Implementation Plan for Lucene

## Executive Summary

This document outlines how to implement a single-layer (flat) HNSW graph in Apache Lucene, based on the findings from the paper "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" (arXiv:2412.01940).

**Key Finding from Paper:** For high-dimensional vectors (d ≥ 32), the hierarchical structure in HNSW provides no performance benefit and wastes ~38% memory during index construction. A flat graph achieves identical latency and recall.

## Current Implementation Analysis

### Level Assignment Mechanism

**Location:** [HnswGraphBuilder.java:482-488](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java#L482-L488)

```java
private static int getRandomGraphLevel(double ml, SplittableRandom random) {
  double randDouble;
  do {
    randDouble = random.nextDouble(); // avoid 0 value, as log(0) is undefined
  } while (randDouble == 0.0);
  return ((int) (-log(randDouble) * ml));
}
```

**ml calculation:** [HnswGraphBuilder.java:154](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java#L154)
```java
this.ml = M == 1 ? 1 : 1 / Math.log(1.0 * M);
```

For default M=16: `ml = 1/log(16) ≈ 0.36`

This produces a distribution where:
- Most nodes are on level 0 (base layer)
- Progressively fewer nodes on levels 1, 2, 3, etc.
- Entry node is typically on the highest level

### Search Process

**Location:** [HnswGraphSearcher.java:222-233](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphSearcher.java#L222-L233)

```java
int[] findBestEntryPoint(RandomVectorScorer scorer, HnswGraph graph, KnnCollector collector) {
  int currentEp = graph.entryNode();
  if (currentEp == -1 || graph.numLevels() == 1) {  // Already handles flat case!
    return new int[] {currentEp};
  }
  // ... top-down hierarchical search
  for (int level = graph.numLevels() - 1; level >= 1; level--) {
    // Navigate from top to bottom
  }
}
```

**Important:** The code already has a fast path when `numLevels() == 1`!

### Storage Format

**Location:** [Lucene99HnswVectorsWriter.java:470-541](lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsWriter.java#L470-L541)

Graph serialization stores:
- Number of levels
- For each level: node offsets and neighbor lists
- Entry node information

Memory overhead comes from storing multiple levels worth of connections.

## Implementation Approach

### Option 1: Add Configuration Parameter (RECOMMENDED)

Add a boolean flag to force all nodes to level 0, controlled by dimensionality or user configuration.

**Advantages:**
- Backward compatible (can read both flat and hierarchical graphs)
- Simple to implement
- Can be toggled per-index based on vector dimensionality
- Minimal code changes

**Changes Required:**

#### 1. Add Configuration to HnswGraphBuilder

```java
// In HnswGraphBuilder.java
private final boolean flatMode;  // New field

protected HnswGraphBuilder(
    RandomVectorScorerSupplier scorerSupplier,
    int M,
    int beamWidth,
    long seed,
    int graphSize,
    boolean flatMode) {  // New parameter
  // ... existing code
  this.flatMode = flatMode;
}

private int getRandomGraphLevel(double ml, SplittableRandom random) {
  if (flatMode) {
    return 0;  // Force all nodes to level 0
  }
  // ... existing logic
}
```

#### 2. Update Factory Methods

```java
// Add new factory methods
public static HnswGraphBuilder create(
    RandomVectorScorerSupplier scorerSupplier,
    int M,
    int beamWidth,
    long seed,
    boolean flatMode) throws IOException {
  return new HnswGraphBuilder(scorerSupplier, M, beamWidth, seed, -1, flatMode);
}

// Auto-detect based on dimensionality
public static HnswGraphBuilder createWithAutoDetection(
    RandomVectorScorerSupplier scorerSupplier,
    int M,
    int beamWidth,
    long seed,
    int vectorDimension) throws IOException {
  boolean useFlatMode = vectorDimension >= 32;  // Based on paper's findings
  return create(scorerSupplier, M, beamWidth, seed, useFlatMode);
}
```

#### 3. Adjust Connection Limits for Flat Mode

```java
// In HnswGraphBuilder.java:addDiverseNeighbors()
int maxConnOnLevel = level == 0 ? M * 2 : M;

// Should become:
int maxConnOnLevel = (level == 0 || flatMode) ? M * 2 : M;
```

Since all nodes are on level 0 in flat mode, they should all get M*2 connections (more neighbors = better connectivity).

#### 4. Expose Configuration in Lucene99HnswVectorsFormat

```java
// In Lucene99HnswVectorsFormat.java
public Lucene99HnswVectorsFormat(int M, int beamWidth, boolean flatMode) {
  // ... existing validation
  this.flatMode = flatMode;
}

// Auto-detection constructor
public Lucene99HnswVectorsFormat(int M, int beamWidth, int vectorDimension) {
  this(M, beamWidth, vectorDimension >= 32);
}
```

#### 5. Storage Format (No Changes Needed!)

The current storage format already supports flat graphs naturally:
- When all nodes are on level 0, `numLevels()` returns 1
- Only level 0 connections are written
- Entry node is any node on level 0
- No format version change required!

### Option 2: Dimension-Based Auto-Detection (ADVANCED)

Automatically switch between flat and hierarchical based on actual vector dimensionality.

**Implementation:**
```java
// In Lucene99HnswVectorsWriter.java
private boolean shouldUseFlatMode(FieldInfo fieldInfo) {
  return fieldInfo.getVectorDimension() >= 32;
}

// When building graph:
boolean flatMode = shouldUseFlatMode(fieldInfo);
HnswGraphBuilder builder = HnswGraphBuilder.create(
    scorerSupplier, M, beamWidth, seed, flatMode);
```

### Option 3: Hybrid Approach with Threshold

Use a configuration threshold to enable flat mode per field.

**In IndexWriterConfig or similar:**
```java
public static final int DEFAULT_FLAT_HNSW_DIMENSION_THRESHOLD = 32;

private int flatHnswDimensionThreshold = DEFAULT_FLAT_HNSW_DIMENSION_THRESHOLD;

public void setFlatHnswDimensionThreshold(int threshold) {
  this.flatHnswDimensionThreshold = threshold;
}
```

## Code Changes Summary

### Files to Modify

1. **[HnswGraphBuilder.java](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java)**
   - Add `flatMode` field
   - Modify `getRandomGraphLevel()` to return 0 when flatMode=true
   - Update constructors and factory methods
   - Adjust connection limits for flat mode

2. **[Lucene99HnswVectorsFormat.java](lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsFormat.java)**
   - Add `flatMode` or dimension threshold configuration
   - Update constructors

3. **[Lucene99HnswVectorsWriter.java](lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsWriter.java)**
   - Pass flatMode to HnswGraphBuilder.create()

4. **[package-info.java](lucene/core/src/java/org/apache/lucene/util/hnsw/package-info.java)**
   - Update documentation to explain flat mode option

### Files That DON'T Need Changes

1. **HnswGraphSearcher.java** - Already handles `numLevels() == 1` efficiently
2. **OnHeapHnswGraph.java** - Already supports single-level graphs
3. **Lucene99HnswVectorsReader.java** - Already reads single-level graphs
4. **Storage format** - No codec version bump needed!

## Performance and Memory Implications

### Expected Benefits (from paper)

**For high-dimensional vectors (d ≥ 32):**
- ✅ **~38% memory savings** during index construction
- ✅ **Identical recall** to hierarchical HNSW
- ✅ **Identical or better latency** (no overhead from level traversal)
- ✅ **Simpler code paths** in search

**For low-dimensional vectors (d < 32):**
- ⚠️ May have **worse performance** - hierarchy provides benefit
- Should keep hierarchical mode for low dimensions

### Memory Analysis

**Current hierarchical HNSW memory usage:**
```
Total = Level 0 connections + Level 1 connections + ... + Level N connections
      ≈ (N nodes * M * 2) + (N/e nodes * M) + (N/e² nodes * M) + ...
      ≈ N * M * 2 * (1 + 1/(2e) + 1/(2e²) + ...)
      ≈ N * M * 2.3  (approximately)
```

**Flat mode memory usage:**
```
Total = Level 0 connections only
      = N nodes * M * 2
```

**Savings:** `(2.3 - 2.0) / 2.3 ≈ 13%` in graph structure, but the paper reports ~38% overall due to additional overhead in multi-level bookkeeping.

### Search Performance

**Hierarchical search complexity:** O(log N) expected hops
**Flat search complexity:** O(log N) expected hops via hub highways

The paper's "Hub Highway Hypothesis" explains why flat performs equally:
- High-dimensional spaces naturally create hub nodes
- Hubs preferentially connect to other hubs
- This forms "highways" that enable fast traversal
- Same effect as hierarchical layers, but emergent

## Testing Strategy

### Unit Tests

1. **Test flat mode produces single-level graph**
```java
@Test
public void testFlatModeProducesSingleLevel() {
  HnswGraphBuilder builder = HnswGraphBuilder.create(
      scorerSupplier, M, beamWidth, seed, true);  // flatMode=true
  // Add nodes
  assertEquals(1, builder.getGraph().numLevels());
}
```

2. **Test flat vs hierarchical recall equivalence**
```java
@Test
public void testFlatVsHierarchicalRecall() {
  // Build both graphs with same data
  // Compare recall@k for various k
  // Should be within 0.1% for d >= 32
}
```

3. **Test memory usage**
```java
@Test
public void testFlatModeMemorySavings() {
  // Build flat and hierarchical graphs
  // Measure ramBytesUsed()
  // Verify flat uses less memory
}
```

### Benchmark Tests

Create benchmarks for:
1. Index construction time
2. Index construction memory
3. Query latency (p50, p99)
4. Recall@10, Recall@100
5. Different vector dimensions: 8, 16, 32, 64, 128, 256, 512, 768, 1536

## Migration Path

### Backward Compatibility

✅ **Fully backward compatible** - No storage format changes needed!

- Old indices with hierarchical graphs continue to work
- New indices can be created as flat or hierarchical
- Codec version doesn't need to change
- Reader automatically detects `numLevels == 1` and optimizes

### Upgrade Procedure

1. **Default behavior:** Keep hierarchical mode by default initially
2. **Opt-in:** Users can enable flat mode via configuration
3. **Future default:** After validation, make flat mode default for d ≥ 32
4. **Reindexing:** Users can reindex to get memory benefits

## Configuration API Design

### Simple Boolean Flag (Phase 1)

```java
Lucene99HnswVectorsFormat format =
    new Lucene99HnswVectorsFormat(16, 100, true);  // M, beamWidth, flatMode
```

### Dimension Threshold (Phase 2)

```java
Lucene99HnswVectorsFormat format =
    new Lucene99HnswVectorsFormat(16, 100, 32);  // Auto flat mode if dim >= 32
```

### Per-Field Configuration (Phase 3)

```java
// In IndexWriterConfig or FieldType
fieldType.setVectorHnswMode(HnswMode.FLAT);
fieldType.setVectorHnswMode(HnswMode.HIERARCHICAL);
fieldType.setVectorHnswMode(HnswMode.AUTO);  // Based on dimension
```

## Risks and Considerations

### 1. Performance Regression for Low-Dimensional Data

**Risk:** Flat mode may hurt performance for d < 32
**Mitigation:** Only enable for d ≥ 32, make it configurable

### 2. Paper is Recent and Unvalidated

**Risk:** Community hasn't validated the findings yet
**Mitigation:**
- Implement as opt-in feature first
- Run comprehensive benchmarks on Lucene's specific workloads
- Gather feedback before making it default

### 3. Different Workload Characteristics

**Risk:** Lucene's use cases may differ from paper's benchmarks
**Mitigation:**
- Test on real-world Lucene datasets
- Test with different similarity functions (L2, cosine, dot product)
- Test with filtered queries (Lucene specialty)

### 4. Hub Formation May Vary

**Risk:** Hub formation depends on data distribution
**Mitigation:**
- Test on diverse datasets (text embeddings, image embeddings, etc.)
- Monitor recall and latency metrics

## Next Steps

### Phase 1: Proof of Concept (1-2 weeks)
1. ✅ Understand current implementation
2. ✅ Identify required changes
3. Implement basic flat mode flag in HnswGraphBuilder
4. Add unit tests
5. Run basic benchmarks

### Phase 2: Integration (2-3 weeks)
1. Add configuration to Lucene99HnswVectorsFormat
2. Implement dimension-based auto-detection
3. Comprehensive testing
4. Performance benchmarking on real datasets

### Phase 3: Validation (4-6 weeks)
1. Community testing and feedback
2. Performance comparison with hierarchical mode
3. Documentation and examples
4. Blog post explaining the feature

### Phase 4: Default Behavior (Future)
1. After sufficient validation
2. Make flat mode default for high-dimensional vectors
3. Update migration guides

## References

1. Paper: "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" (arXiv:2412.01940)
2. Original HNSW paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2018)
3. FlatNav implementation: https://github.com/BlaiseMuhirwa/flatnav

## Appendix: Key Code Locations

- Level assignment: [HnswGraphBuilder.java:482](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java#L482)
- ml calculation: [HnswGraphBuilder.java:154](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java#L154)
- Search entry point: [HnswGraphSearcher.java:225](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphSearcher.java#L225)
- Graph writing: [Lucene99HnswVectorsWriter.java:470](lucene/core/src/java/org/apache/lucene/codecs/lucene99/Lucene99HnswVectorsWriter.java#L470)
- Connection limits: [HnswGraphBuilder.java:359](lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java#L359)
