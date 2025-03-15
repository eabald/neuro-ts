import { Decimal } from 'decimal.js';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Neuron } from './neuron';

describe('Neuron', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('should initialize with random bias', () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const neuron = new Neuron(3, (x) => x);

    const bias = neuron.getBias();
    expect(bias.toNumber()).toBe(0.5 * 2 - 1);
  });

  it('should set and get weights correctly', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setWeights([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    expect(neuron.getWeights()).toEqual([0.1, 0.2, 0.3].map(val => new Decimal(val)));
  });

  it('should set and get bias correctly', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setBias(new Decimal(0.5));
    expect(neuron.getBias().toNumber()).toBe(new Decimal(0.5).toNumber());
  });

  it('should throw error if input size does not match weights length', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setWeights([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    expect(() => neuron.forward([0.1, 0.2].map(val => new Decimal(val)))).toThrowError('Input size must match number of weights');
  });

  it('should compute forward pass correctly', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setWeights([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    neuron.setBias(new Decimal(0.5));
    const output = neuron.forward([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    expect(output.toNumber()).toEqual((new Decimal(0.64)).toNumber());
  });

  it('should return correct weights after backward pass', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setWeights([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    neuron.setBias(new Decimal(0.5));
    const weights = neuron.backward(new Decimal(0.1), new Decimal(0.01));
    expect(weights.map(weight => weight.toNumber())).toEqual([new Decimal(0.01005), new Decimal(0.02005), new Decimal(0.03005)].map(val => val.toNumber()));
  });

  it('should return correct bias after backward pass', () => {
    const neuron = new Neuron(3, (x) => x);
    neuron.setWeights([0.1, 0.2, 0.3].map(val => new Decimal(val)));
    neuron.setBias(new Decimal(0.5));
    neuron.backward(new Decimal(0.1), new Decimal(0.01));
    expect(neuron.getBias().toNumber()).toEqual((new Decimal(0.5005)).toNumber());
  });
});
