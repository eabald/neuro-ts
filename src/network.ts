import { all, create } from 'mathjs';
import { softmax } from '.';
import { Layer } from "./layer";

const math = create(all);

export class Network {

  constructor(private layers: Layer[], data?: Buffer) {
    if (data) {
      this.importWeightsAndBiases(data);
    }
  }

  forward(inputs: number[]): number[] {
    return this.layers.reduce((prev, layer) => layer.forward(prev), inputs);
  }

  train(inputs: number[], targets: number[], learningRate: number): void {
    const outputs = this.forward(inputs);
    let errors = targets.map((target, i) => target - outputs[i]);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const backward = this.layers[i].backward(errors, learningRate);
      const mapped = backward.map(val => math.sum(val));
      const flattened = mapped.flat();
      errors = flattened;
    }
  }

  predict(inputs: number[]): number[] {
    return softmax(this.forward(inputs));
  }

  getWeights(): number[][][] {
    return this.layers.map(layer => layer.getWeights());
  }

  setWeights(weights: number[][][]): void {
    this.layers.forEach((layer, i) => layer.setWeights(weights[i]));
  }

  getBiases(): number[][] {
    return this.layers.map(layer => layer.getBiases());
  }

  setBiases(biases: number[][]): void {
    this.layers.forEach((layer, i) => layer.setBiases(biases[i]));
  }

  exportWeightsAndBiases(): Buffer {
    const weights = this.getWeights();
    const biases = this.getBiases();
    const weightsBuffer = Buffer.from(JSON.stringify(weights), 'utf-8');
    const biasesBuffer = Buffer.from(JSON.stringify(biases), 'utf-8');
    return Buffer.concat([weightsBuffer, biasesBuffer]);
  }

  importWeightsAndBiases(data: Buffer): void {
    const [weightsData, biasesData] = data.toString('utf-8').split(',');
    const weights = JSON.parse(weightsData);
    const biases = JSON.parse(biasesData);
    this.setWeights(weights);
    this.setBiases(biases);
  }
}
