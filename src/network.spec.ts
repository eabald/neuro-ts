import Decimal from 'decimal.js';
import { beforeEach, describe, expect, it } from 'vitest';
import { Layer } from './layer';
import { Network } from './network';
import { Neuron } from './neuron';

describe('Network', () => {
  let network: Network;
  let layers: Layer[];

  beforeEach(() => {
    const neurons1 = [
      new Neuron(3, (x) => x),
      new Neuron(3, (x) => x),
    ];
    const neurons2 = [
      new Neuron(2, (x) => x),
    ];
    layers = [
      new Layer(neurons1),
      new Layer(neurons2),
    ];
    network = new Network(layers);

    layers[0].setWeights([
      [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)],
      [new Decimal(0.4), new Decimal(0.5), new Decimal(0.6)],
    ]);
    layers[0].setBiases([new Decimal(0.1), new Decimal(0.2)]);
    layers[1].setWeights([
      [new Decimal(0.7), new Decimal(0.8)],
    ]);
    layers[1].setBiases([new Decimal(0.3)]);
  });

  it('should forward inputs correctly', () => {

    const inputs = [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)];
    const outputs = network.forward(inputs);
    expect(outputs.map(output => output.toNumber())).toEqual([0.884]);
  });

  it('should train the network correctly', () => {
    const inputs = [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)];
    const targets = [new Decimal(1.0)];
    const learningRate = new Decimal(0.01);

    network.train(inputs, targets, learningRate);

    const updatedWeights1 = layers[0].getWeights();
    const updatedBiases1 = layers[0].getBiases();
    const updatedWeights2 = layers[1].getWeights();
    const updatedBiases2 = layers[1].getBiases();

    expect(updatedWeights1).not.toEqual([
      [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)],
      [new Decimal(0.4), new Decimal(0.5), new Decimal(0.6)],
    ]);
    expect(updatedBiases1).not.toEqual([new Decimal(0.1), new Decimal(0.2)]);
    expect(updatedWeights2).not.toEqual([
      [new Decimal(0.7), new Decimal(0.8)],
    ]);
    expect(updatedBiases2).not.toEqual([new Decimal(0.3)]);
  });

  it('should predict outputs correctly', () => {
    const inputs = [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)];
    const outputs = network.predict(inputs);
    expect(outputs.map(output => output.toNumber())).toEqual([0.884]);
  });
});
