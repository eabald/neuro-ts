import Decimal from 'decimal.js';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Layer } from './layer';
import { Neuron } from './neuron';

describe('Layer', () => {
  let neurons: Neuron[];
  let layer: Layer;

  beforeEach(() => {
    neurons = [
      new Neuron(3, (x) => x),
      new Neuron(3, (x) => x),
      new Neuron(3, (x) => x),
    ];
    layer = new Layer(neurons);
  });

  it('should forward inputs correctly', () => {
    neurons.forEach(neuron => {
      neuron.setWeights([new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)]);
      neuron.setBias(new Decimal(0.5));
    });
    const inputs = [new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)];
    const outputs = layer.forward(inputs);
    outputs.forEach(output => {
      expect(output.toNumber()).toBeCloseTo(0.64, 5);
    });
  });

  it('should get weights correctly', () => {
    neurons.forEach(neuron => {
      neuron.setWeights([new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)]);
    });
    const weights = layer.getWeights();
    weights.forEach((weights) => {
      expect(weights.map(weight => weight.toNumber())).toEqual([0.1, 0.2, 0.3]);
    });
  });

  it('should set weights correctly', () => {
    const newWeights = [
      [new Decimal(0.4), new Decimal(0.5), new Decimal(0.6)],
      [new Decimal(0.7), new Decimal(0.8), new Decimal(0.9)],
      [new Decimal(1.0), new Decimal(1.1), new Decimal(1.2)],
    ];
    layer.setWeights(newWeights);
    expect(layer.getWeights()).toEqual(newWeights);
  });

  it('should get biases correctly', () => {
    neurons.forEach(neuron => {
      neuron.setBias(new Decimal(0.5));
    });
    const biases = layer.getBiases();
    expect(biases.map(bias => bias.toNumber())).toEqual([0.5, 0.5, 0.5]);
  });

  it('should set biases correctly', () => {
    const newBiases = [new Decimal(0.6), new Decimal(0.7), new Decimal(0.8)];
    layer.setBiases(newBiases);
    neurons.forEach((neuron, i) => {
      const bias = neuron.getBias();
      expect(bias.toNumber()).toEqual(newBiases[i].toNumber());
    });
  });

  it('should perform backward pass correctly', () => {
    vi.spyOn(neurons[0], 'getWeights').mockReturnValue([new Decimal(0.1), new Decimal(0.2), new Decimal(0.3)]);
    const backwardSpyOn = vi.spyOn(neurons[0], 'backward');
    const errors = [new Decimal(0.1), new Decimal(-0.2)];
    const learningRate = new Decimal(0.05);

    layer.backward(errors, learningRate);

    expect(backwardSpyOn).toHaveBeenCalledTimes(1);
    expect(backwardSpyOn).toBeCalledWith(new Decimal(-0.03), learningRate);
  });
});
