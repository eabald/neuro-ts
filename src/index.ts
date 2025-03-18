import { Presets, SingleBar } from 'cli-progress';
import { writeFileSync } from 'fs';
import { Layer } from "./layer";
import { Network } from "./network";
import { Neuron } from "./neuron";
import { open } from "./open";

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function softmax(x: number[]): number[] {
  const expValues = x.map(value => Math.exp(value));
  const sumExpValues = expValues.reduce((a, b) => a + b, 0);
  return expValues.map(value => value / sumExpValues);
}

function Main() {
  console.log("Opening training and testing data...");
  const trainingData = open("data/train-images-idx3-ubyte/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
  const testingData = open("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");

  console.log("Initializing network...");
  const network = new Network([
    new Layer(Array.from({ length: 784 }, () => new Neuron(784, sigmoid))),
    new Layer(Array.from({ length: 16 }, () => new Neuron(784, sigmoid))),
    new Layer(Array.from({ length: 16 }, () => new Neuron(16, sigmoid))),
    new Layer(Array.from({ length: 10 }, () => new Neuron(16, sigmoid)))
  ]);

  const bar = new SingleBar({}, Presets.shades_classic);

  console.log("Starting training...");
  for (let epoch = 0; epoch < 1; epoch++) {
    console.log(`Epoch ${epoch + 1}`);
    bar.start(trainingData.length, 0);
    let counter = 0;
    for (const data of trainingData) {
      counter++;
      const inputs = data.data.map(x => x);
      const targets = Array.from({ length: 10 }, (_, i) => i === Number(data.label) ? 1 : 0);
      bar.update(counter);
      network.train(inputs, targets, 0.1);
    }
    bar.stop();
  }

  writeFileSync("network", network.exportWeightsAndBiases());

  console.log("Training completed. Starting testing...");
  let correct = 0;
  bar.start(trainingData.length, 0);
  let counter = 0;
  for (const data of testingData) {
    counter++;
    const inputs = data.data.map(x => x);
    const prediction = network.predict(inputs);
    const actual = Number(data.label);
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    bar.update(counter);
    if (maxIndex === actual) {
      correct++;
    }
  }
  console.log(`Accuracy: ${(correct / testingData.length * 100).toFixed(2)}%`);
}

Main();
