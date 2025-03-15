import Decimal from "decimal.js";
import { Neuron } from "./neuron";

export class Layer {
  constructor(private neurons: Neuron[]) { }

  forward(inputs: Decimal[]): Decimal[] {
    return this.neurons.map(neuron => neuron.forward(inputs));
  }

  backward(errors: Decimal[], learningRate: Decimal): Decimal[][] {
    return this.neurons.map((neuron) => {
      const weights = neuron.getWeights();
      const error = errors.reduce((acc, val, j) => {
        return acc.plus(val.times(weights[j]));
      }, new Decimal(0));
      return neuron.backward(error, learningRate);
    });
  }

  getWeights(): Decimal[][] {
    return this.neurons.map(neuron => neuron.getWeights());
  }

  setWeights(weights: Decimal[][]): void {
    let index = 0;
    for (const neuron of this.neurons) {
      neuron.setWeights(weights[index++]);
    }
  }

  getBiases(): Decimal[] {
    return this.neurons.map(neuron => neuron.getBias());
  }

  setBiases(biases: Decimal[]): void {
    this.neurons.forEach((neuron, i) => {
      neuron.setBias(biases[i]);
    });
  }
}
