# SKETCH-STYLUS
![alt text](public/Untitled-5_page-0001.jpg)

## Introduction

This is a hand written and optimized machine learning library written in Rust using Arbitrum's Stylus [SDK](https://docs.arbitrum.io/stylus/stylus-quickstart).

This library implements the following:

- K Nearest Neighbours
- Logistic Regression
- Digit Recognizer ML Model
- Machine Learning Math library

## Feautures

- For avoiding the loss of precision in weights and biases, scaling has been implemented.
- Minimal use of std library.
- The size of WASM contract and deployment gas for each is listed below.

### KNN

> Uncompressed size - 24.4 KB
>
> Compressed size - 8.9 KB
>
> Deployment gas - 6881292
>
>Exported ABI - 
>``` bash
>interface IKNN {
>    function trainPredict() external;
>
>    function setK(uint256 val) external;
>
>    function getK() external view returns (uint256);
>}
>```

### Logistic Regression

> Uncompressed size - 55.8 KB
>
> Compressed size - 17.5 KB
>
> Deployment gas -
>
>

### ML-math

> Uncompressed size - 45.2 KB
>
> Compressed size - 13.1 KB
>
> Deployment gas - 10062745
>
>Exported ABI - 
>``` bash
>interface IMlMath {
>    function reluDerive(int128[][] memory m) external pure returns (int128[][] memory);
>
>    function elementSumRow(int128[][] memory z) external pure returns (int128[][] memory);
>
>    function scalarDiv(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory);
>
>    function elementwiseMul(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);
>
>    function transpose(int128[][] memory ori) external pure returns (int128[][] memory);
>
>    function softmax(int128[][] memory z) external pure returns (int128[][] memory);
>
>    function relu(int128[][] memory z) external pure returns (int128[][] memory);
>
>    function dotProduct(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);
>
>    function sum(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);
>
>    function oneHot(int128[][] memory y) external pure returns (int128[][] memory);
>
>    function scalarMul(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory);
>}
> ```

### Digit Recognizer

> Uncompressed size - 52 KB
>
> Compressed size - 16.5 KB
>
> Deployment gas - 12623680
>
>Exported ABI - 
>``` bash
>interface ITraining {
>    function trainPredict(int128[][] memory x_train, int128[][] memory y_train) external returns (bool);
>}
>```
