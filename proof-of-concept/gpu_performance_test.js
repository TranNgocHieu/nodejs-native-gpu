const tensor = require('./native.node');

// Performance test function - matmul operator on all adapters
async function runPerformanceTest() {
  console.log('=== GPU Adapter Performance Test ===\n');

  try {
    // 1. Get all available adapters
    console.log('Available GPU adapters:');
    const adapters = tensor.getAdapters();

    if (!adapters || adapters.length === 0) {
      console.log('No GPU adapters available!');
      return;
    }

    adapters.forEach((adapter, index) => {
      console.log(`${index}: ${adapter.name} (${adapter.backend})`);
    });
    console.log('');

    // 2. Create test matrices - HEAVY load for performance measurement
    const matrixSize = 1024; // 1024x1024 matrices (4x bigger load)
    console.log(
      `Test matrix size: ${matrixSize}x${matrixSize} (${((matrixSize * matrixSize) / 1000000).toFixed(1)}M elements)`,
    );
    console.log('');

    const results = [];

    // 3. Go through all adapters and measure performance
    for (let i = 0; i < adapters.length; i++) {
      const adapter = adapters[i];
      console.log(`\n--- Adapter ${i}: ${adapter.name} ---`);

      try {
        // GPU context initialization for this adapter
        const contextId = `adapter_${i}`;
        const success = tensor.initGpu(i, contextId);

        if (!success) {
          console.log(`FAILED: Adapter ${i} initialization failed`);
          results.push({
            adapter: i,
            name: adapter.name,
            backend: adapter.backend,
            status: 'INIT_FAILED',
            time: null,
            throughput: null,
          });
          continue;
        }

        console.log(`SUCCESS: GPU context initialized: ${contextId}`);

        // Random matrix generation for this context
        const matrixA = tensor.randomUniform([matrixSize, matrixSize], 0.0, 1.0, contextId);
        const matrixB = tensor.randomUniform([matrixSize, matrixSize], 0.0, 1.0, contextId);
        const matrixC = tensor.randomUniform([matrixSize, matrixSize], 0.0, 1.0, contextId);

        // Warmup runs (GPU pipeline initialization)
        console.log('Warmup runs...');
        for (let warmup = 0; warmup < 3; warmup++) {
          const warmupResult = tensor.matmul(matrixA, matrixB, contextId);
          tensor.dispose(warmupResult);
        }

        // INTENSIVE performance measurement - multiple runs and combined operations
        const testRuns = 10; // 10 runs for more accurate average
        const times = [];

        console.log(`Intensive performance measurement (${testRuns} runs + combined operations)...`);

        for (let run = 0; run < testRuns; run++) {
          const startTime = performance.now();

          // COMBINED GPU OPERATIONS - heavier load
          const result1 = tensor.matmul(matrixA, matrixB, contextId);
          const result2 = tensor.matmul(matrixB, matrixC, contextId);
          const result3 = tensor.add(result1, result2, contextId);
          const result4 = tensor.multiply(result3, matrixA, contextId);
          const finalResult = tensor.matmul(result4, matrixB, contextId);

          const endTime = performance.now();

          // Cleanup all results
          tensor.dispose(result1);
          tensor.dispose(result2);
          tensor.dispose(result3);
          tensor.dispose(result4);
          tensor.dispose(finalResult);

          const runTime = endTime - startTime;
          times.push(runTime);
          console.log(`  Run ${run + 1}: ${runTime.toFixed(2)}ms`);
        }

        // Average time calculation
        const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);

        // Throughput calculation (GFLOPS) - combined operations calculation
        // 3x matmul (2 * n^3 each) + 1x add (n^2) + 1x multiply (n^2)
        const matmulOps = 3 * 2 * Math.pow(matrixSize, 3);
        const addOps = Math.pow(matrixSize, 2);
        const mulOps = Math.pow(matrixSize, 2);
        const totalOperations = matmulOps + addOps + mulOps;
        const throughputGFLOPS = totalOperations / (avgTime / 1000) / 1e9;

        console.log(
          `STATS: Average time: ${avgTime.toFixed(2)}ms (min: ${minTime.toFixed(2)}ms, max: ${maxTime.toFixed(2)}ms)`,
        );
        console.log(`PERF: Throughput: ${throughputGFLOPS.toFixed(2)} GFLOPS`);
        console.log(
          `OPS: Combined operations: 3x matmul + add + multiply = ${(totalOperations / 1e9).toFixed(2)}B ops`,
        );

        results.push({
          adapter: i,
          name: adapter.name,
          backend: adapter.backend,
          status: 'SUCCESS',
          time: avgTime,
          minTime: minTime,
          maxTime: maxTime,
          throughput: throughputGFLOPS,
          totalOperations: totalOperations,
          times: times,
        });

        // Cleanup matrices for this adapter
        tensor.dispose(matrixA);
        tensor.dispose(matrixB);
        tensor.dispose(matrixC);
      } catch (error) {
        console.log(`ERROR: Error with adapter ${i}: ${error.message}`);
        results.push({
          adapter: i,
          name: adapter.name,
          backend: adapter.backend,
          status: 'ERROR',
          error: error.message,
          time: null,
          throughput: null,
        });
      }
    }

    // 4. Summary report
    console.log('\n' + '='.repeat(60));
    console.log('PERFORMANCE SUMMARY');
    console.log('='.repeat(60));

    // Sort successful tests by performance
    const successfulTests = results.filter((r) => r.status === 'SUCCESS');
    successfulTests.sort((a, b) => b.throughput - a.throughput);

    if (successfulTests.length > 0) {
      console.log('\nPERFORMANCE RESULTS:');
      successfulTests.forEach((result, index) => {
        const variance = (((result.maxTime - result.minTime) / result.time) * 100).toFixed(1);
        console.log(`${index + 1}. ${result.name}`);
        console.log(`    Backend: ${result.backend}`);
        console.log(`    Average time: ${result.time.toFixed(2)}ms (±${variance}%)`);
        console.log(`    Throughput: ${result.throughput.toFixed(2)} GFLOPS`);
        console.log(`    Operations: ${(result.totalOperations / 1e9).toFixed(2)}B ops`);
        console.log('');
      });
    }

    // Failed adapters
    const failedTests = results.filter((r) => r.status !== 'SUCCESS');
    if (failedTests.length > 0) {
      console.log('FAILED ADAPTERS:');
      failedTests.forEach((result) => {
        console.log(`   ${result.name} (${result.backend}): ${result.status}`);
        if (result.error) {
          console.log(`      Error: ${result.error}`);
        }
      });
    }

    // Save complete results to JSON
    const resultData = {
      timestamp: new Date().toISOString(),
      testConfig: {
        matrixSize: matrixSize,
        testRuns: 10,
        operations: ['3x matmul', 'add', 'multiply'],
        description: 'Intensive combined GPU operations',
      },
      results: results,
    };

    require('fs').writeFileSync('./gpu_performance_results.json', JSON.stringify(resultData, null, 2));

    console.log('\nDetailed results saved to: gpu_performance_results.json');
  } catch (error) {
    console.error('Error during performance testing:', error);
  }
}

// Start test
if (require.main === module) {
  console.log('Starting GPU adapter performance test...\n');
  runPerformanceTest().catch(console.error);
}

module.exports = { runPerformanceTest };
