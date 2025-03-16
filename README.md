# NeuroTS

NeuroTS is a TypeScript-based neural network designed for educational purposes. It provides a simple implementation of neural networks, including layers, neurons, and activation functions.

## Features

- Simple neural network implementation
- Support for different activation functions
- Training and testing with MNIST dataset

## Installation

To install the dependencies, run:

```sh
npm install
```

## Usage

To build the project, run:

```sh
npm run build
```

To start the project, run:

```sh
npm start
```

To run tests, use:

```sh
npm test
```

## Project Structure

- `src/`: Contains the source code of the project
  - `index.ts`: Entry point of the application
  - `layer.ts`: Implementation of the Layer class
  - `network.ts`: Implementation of the Network class
  - `neuron.ts`: Implementation of the Neuron class
  - `open.ts`: Function to open and read the dataset files
  - `types/`: Contains TypeScript type definitions
  - `tests/`: Contains the test files
- `data/`: Contains the MNIST dataset files

## Dependencies

~~`decimal.js`: Library for arbitrary-precision decimal arithmetic~~
`math.js`: better performance than decimal

## License

This project is licensed under the MIT License.
