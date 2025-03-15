import Decimal from "decimal.js";
import { Layer } from "./layer";
import { Network } from "./network";
import { Neuron } from "./neuron";
import { open } from "./open";

export function sigmoid(x: Decimal): Decimal {
  return Decimal.exp(x.neg().toNumber()).plus(1).pow(-1);
}

export function softmax(x: Decimal): Decimal {
  return Decimal.exp(x).div(Decimal.sum(Decimal.exp(x)));
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
    new Layer(Array.from({ length: 10 }, () => new Neuron(16, softmax)))
  ]);

  console.log("Starting training...");
  for (let epoch = 0; epoch < 10; epoch++) {
    console.log(`Epoch ${epoch + 1}`);
    let counter = 0;
    for (const data of trainingData) {
      counter++;
      console.log(`Training data ${counter} of ${trainingData.length}`);
      const inputs = data.data.map(x => new Decimal(x));
      const targets = Array.from({ length: 10 }, (_, i) => i === Number(data.label) ? new Decimal(1) : new Decimal(0));
      network.train(inputs, targets, new Decimal(0.1));
    }
  }

  console.log("Training completed. Starting testing...");
  let correct = 0;
  for (const data of testingData) {
    const inputs = data.data.map(x => new Decimal(x));
    const prediction = network.predict(inputs).map(x => x.toNumber());
    const actual = Number(data.label);
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    if (maxIndex === actual) {
      correct++;
    }
  }
  console.log(`Accuracy: ${(correct / testingData.length * 100).toFixed(2)}%`);
}

Main();
