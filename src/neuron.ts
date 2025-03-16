import { all, create } from 'mathjs';

const math = create(all);

export class Neuron {
  weights: number[];
  bias: number;
  activation: (x: number) => number;

  constructor(inputLength: number, activation: (x: number) => number) {
    this.bias = Math.random() * 2 - 1;
    this.weights = Array.from({ length: inputLength }, () => Math.random() * 2 - 1);
    this.activation = activation;
  }

  forward(inputs: number[]): number {
    if (inputs.length !== this.weights.length) {
      throw new Error(`Input size must match number of weights, input size: ${inputs.length}, number of weights: ${this.weights.length}`);
    }
    const sum = math.add(math.dot(inputs, this.weights), this.bias);
    return this.activation(sum);
  }

  backward(error: number, learningRate: number): number[] {
    const gradient = error * this.activation(this.bias);
    this.weights = math.add(this.weights, math.multiply(gradient, learningRate)) as number[];
    this.bias += gradient * learningRate;
    const updatedWeights = math.multiply(this.weights, error) as number[];
    return updatedWeights;
  }

  getWeights(): number[] {
    return this.weights;
  }

  setWeights(weights: number[]): void {
    this.weights = weights;
  }

  setBias(bias: number): void {
    this.bias = bias;
  }

  getBias(): number {
    return this.bias;
  }
}
