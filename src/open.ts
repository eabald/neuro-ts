import { readFileSync } from 'fs';

export function open(dataPath: string, labelsPath: string): {
  label: string;
  data: number[];
}[] {
  const dataFileBuffer = readFileSync(dataPath);
  const labelFileBuffer = readFileSync(labelsPath);
  const pixelValues = [];

  for (let image = 0; image <= 59999; image++) {
    const pixels = [];

    for (let x = 0; x <= 27; x++) {
      for (let y = 0; y <= 27; y++) {
        pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 15]);
      }
    }

    const label = JSON.stringify(labelFileBuffer[image + 8]);
    const data = pixels;
    const imageData = {
      label,
      data
    };

    pixelValues.push(imageData);
  }

  return pixelValues;
}
