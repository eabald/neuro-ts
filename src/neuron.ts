import { Decimal } from 'decimal.js';

export class Neuron {
  weights: Decimal[];
  bias: Decimal;
  activation: (x: Decimal) => Decimal;

  constructor(inputLength: number, activation: (x: Decimal) => Decimal) {
    this.bias = new Decimal(Math.random() * 2 - 1);
    this.weights = Array.from({ length: inputLength }, () => new Decimal(Math.random() * 2 - 1));
    this.activation = activation;
  }

  forward(inputs: Decimal[]): Decimal {
    if (inputs.length !== this.weights.length) {
      throw new Error("Input size must match number of weights");
    }
    const sum = inputs.reduce((acc, val, i) => acc.plus(val.times(this.weights[i])), this.bias);
    return this.activation(sum);
  }

  backward(error: Decimal, learningRate: Decimal): Decimal[] {
    const gradient = error.times(this.activation(this.bias));
    this.weights = this.weights.map(weight => weight.plus(gradient.times(learningRate)));
    this.bias = this.bias.plus(gradient.times(learningRate));
    const updatedWeights = this.weights.map(weight => weight.times(error));
    return updatedWeights;
  }

  getWeights(): Decimal[] {
    return this.weights;
  }

  setWeights(weights: Decimal[]): void {
    this.weights = weights;
  }

  setBias(bias: Decimal): void {
    this.bias = bias;
  }

  getBias(): Decimal {
    return this.bias;
  }
}
