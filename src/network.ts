import Decimal from "decimal.js";
import { Layer } from "./layer";

export class Network {

  constructor(private layers: Layer[]) { }

  forward(inputs: Decimal[]): Decimal[] {
    return this.layers.reduce((prev, layer) => layer.forward(prev), inputs);
  }

  train(inputs: Decimal[], targets: Decimal[], learningRate: Decimal): void {
    const outputs = this.forward(inputs);
    let errors = targets.map((target, i) => target.minus(outputs[i]));
    for (let i = this.layers.length - 1; i >= 0; i--) {
      errors = this.layers[i].backward(errors, learningRate).map(val => val.reduce((acc, current) => acc.add(current), new Decimal(0))).flat();
    }
  }

  predict(inputs: Decimal[]): Decimal[] {
    return this.forward(inputs);
  }
}
