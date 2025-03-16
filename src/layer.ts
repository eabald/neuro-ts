import { all, create } from 'mathjs';
import { Neuron } from "./neuron";

const math = create(all);

export class Layer {
  constructor(private neurons: Neuron[]) { }

  forward(inputs: number[]): number[] {
    return this.neurons.map(neuron => neuron.forward(inputs));
  }

  backward(errors: number[], learningRate: number): number[][] {
    return this.neurons.map((neuron) => {
      const weights = neuron.getWeights();
      const avgError = math.mean(errors);
      const error = math.dot(Array.from({ length: weights.length }).map(() => avgError), weights);
      return neuron.backward(error, learningRate);
    });
  }

  getWeights(): number[][] {
    return this.neurons.map(neuron => neuron.getWeights());
  }

  setWeights(weights: number[][]): void {
    let index = 0;
    for (const neuron of this.neurons) {
      neuron.setWeights(weights[index++]);
    }
  }

  getBiases(): number[] {
    return this.neurons.map(neuron => neuron.getBias());
  }

  setBiases(biases: number[]): void {
    this.neurons.forEach((neuron, i) => {
      neuron.setBias(biases[i]);
    });
  }
}
