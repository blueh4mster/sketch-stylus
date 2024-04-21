# SKETCH-STYLUS

<img src="public/Untitled-5_page-0001.jpg" alt="logo" width="448" height="336" style="display:block;margin:0 auto">

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

The k-nearest neighbors (KNN) algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.

- Uncompressed size - 48.4 KB
- Compressed size - 15.6 KB
- Deployment gas - 6881292
- Exported ABI

```javascript
interface IKNN {
   function trainPredict() external;

   function setK(uint256 val) external;

   function getK() external view returns (uint256);
}
```

### Logistic Regression

Logistic regression estimates the probability of an event occurring, such as voted or didn’t vote, based on a given data set of independent variables. This type of statistical model (also known as logit model) is often used for classification and predictive analytics.

- Uncompressed size - 71.1 KB
- Compressed size - 24.0 KB
- Deployment gas -
- Exported ABI

```javascript
interface ILogReg {
    function train(int128[][] memory x_train, int128[][] memory y_train, uint128 iterations, int128 lr) external returns (bool);
}
```

### ML-math

This is a library for mathematical functions required within the different implementations of machine learning models. This includes various functions such as `sigmoid`, `sqrt`, `signum`, `softmax` and other matrix operations such as `matrix_mul`, `dot_product`, `matrix_sub`, `matrix_sum` etc.

- Uncompressed size - 45.2 KB
- Compressed size - 13.1 KB
- Deployment gas - 10062745
- Exported ABI

```javascript
interface IMlMath {
   function reluDerive(int128[][] memory m) external pure returns (int128[][] memory);

   function elementSumRow(int128[][] memory z) external pure returns (int128[][] memory);

   function scalarDiv(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory);

   function elementwiseMul(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);

   function transpose(int128[][] memory ori) external pure returns (int128[][] memory);

   function softmax(int128[][] memory z) external pure returns (int128[][] memory);

   function relu(int128[][] memory z) external pure returns (int128[][] memory);

   function dotProduct(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);

   function sum(int128[][] memory m1, int128[][] memory m2) external pure returns (int128[][] memory);

   function oneHot(int128[][] memory y) external pure returns (int128[][] memory);

   function scalarMul(int128[][] memory mat, int128 scalar) external pure returns (int128[][] memory);
}
```

### Digit Recognizer

Digit Recognition is a computer vision technique to predict the correct digits from pixel values of images. We have employed MNIST dataset for training.

- Uncompressed size - 52 KB
- Compressed size - 16.5 KB
- Deployment gas - 12623680
- Exported ABI

```javascript
interface ITraining {
   function trainPredict(int128[][] memory x_train, int128[][] memory y_train) external returns (bool);
}
```
