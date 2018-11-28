using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace hw
{
    internal class Neuron
    {

    }
    public class NNet
    {
        public double[] InputLayerValues;
        public double[] HiddenNeuronValues;
        public double[] OutputNeuronValues;
        public double[] HiddenNeuronDeltas;
        public double[] OutputNeuronDeltas;

        public double[,] InputToHiddenWeights;
        public double[,] HiddenToOutputWeights;

        public List<Image> InputData;

        public double LearningRate { get; set; }

        private int _inputLayerCount;
        private int _hiddenLayerCount;
        private int _outputLayerCount;
        private double _nu;
        /// <summary>
        /// Neural Network
        /// </summary>
        /// <param name="inputImages"></param>
        /// <param name="hiddenLayerCount"></param>
        /// <param name="outputLayerCount"></param>
        public NNet(List<Image> inputImages, int hiddenLayerCount, int outputLayerCount, double nu)
        {
            _nu = nu;
            InputData = new List<Image>(inputImages);
            AllocateInputData(InputData[0]);
            _hiddenLayerCount = hiddenLayerCount;
            _outputLayerCount = outputLayerCount;

            HiddenNeuronValues = new double[_hiddenLayerCount];
            OutputNeuronValues = new double[_outputLayerCount];
            HiddenNeuronDeltas = new double[_hiddenLayerCount];
            OutputNeuronDeltas = new double[_outputLayerCount];
            InputToHiddenWeights = new double[_hiddenLayerCount, _inputLayerCount];
            HiddenToOutputWeights = new double[_outputLayerCount, _hiddenLayerCount];
        }

        public void Train(int NumberOfSteps)
        {
            var rnd = new Random();

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _inputLayerCount; j++)
                {
                    InputToHiddenWeights[i, j] = rnd.NextDouble() / 100d;
                }
            }
            for (int i = 0; i < _outputLayerCount; i++)
            {
                for (int j = 0; j < _hiddenLayerCount; j++)
                {
                    HiddenToOutputWeights[i, j] = rnd.NextDouble() / 100d;
                }
            }

            var output = new double[_outputLayerCount];

            for (int i = 0; i < NumberOfSteps; i++)
            {
                Console.WriteLine("StartAllocateRandomData");

                var randomizedInput = InputData.OrderBy(x => rnd.Next()).ToList();

                Console.WriteLine("FinishAllocateRandomData");

                foreach (var image in randomizedInput)
                {
                    Array.Clear(output, 0, _outputLayerCount);
                    AllocateInputData(image);
                    output[image.Label] = 1d;

                    ComputeNeuronsHiddenValue();
                    ComputeNeuronsOutputValues();

                    ComputeDerivatives(output);
                    ChangeWeights();
                }
                Console.WriteLine("Step " + i);
                var err = ComputeCrossEntropy();
                if (err < 0.005d)
                {
                    Console.WriteLine("TrainedSuccessfully");
                    return;
                }
                Console.WriteLine(err + " next");
            }
            Console.WriteLine("Trained");
        }

        public void Test(List<Image> image)
        {
            var guessed = 0;

            foreach (var img in image)
            {
                var result = Test(img);

                if (result == img.Label)
                    guessed++;
            }
            Console.WriteLine("Guessed: " + guessed);
            Console.WriteLine("Count: " + image.Count);
            Console.WriteLine("Guessed Rate: " + (double)guessed / image.Count);
        }

        public int Test(Image image)
        {
            AllocateInputData(image);

            ComputeNeuronsHiddenValue();
            ComputeNeuronsOutputValues();

            var max = -1d;
            var maxNum = 0;

            for (int i = 0; i < _outputLayerCount; i++)
            {
                if (OutputNeuronValues[i] > max)
                {
                    max = OutputNeuronValues[i];
                    maxNum = i;
                }
            }

            return maxNum;
        }

        private void AllocateInputData(Image image)
        {
            var count = image.Data.GetLength(0);
            _inputLayerCount = count * count;

            InputLayerValues = new double[_inputLayerCount];

            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    InputLayerValues[i + j * count] = image.Data[i, j] / 255f;
                }
            }
        }

        private double SigmoidActivation(double value)
        {
            return 1 / (1 + Math.Exp(-value));
        }

        private void ComputeNeuronsHiddenValue()
        {
            var sum = 0d;

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                sum = 0d;

                for (int j = 0; j < _inputLayerCount; j++)
                {
                    sum += InputLayerValues[j] * InputToHiddenWeights[i, j];
                }

                HiddenNeuronValues[i] = SigmoidActivation(sum);
            }
        }

        private void ComputeNeuronsOutputValues()
        {
            var sum = 0d;

            var expSum = 0d;

            for (int i = 0; i < _outputLayerCount; i++)
            {
                sum = 0d;

                for (int j = 0; j < _hiddenLayerCount; j++)
                {
                    sum += HiddenNeuronValues[j] * HiddenToOutputWeights[i, j];
                }

                OutputNeuronValues[i] = sum;

                expSum += Math.Exp(sum);
            }

            SoftMaxActivation(expSum);
        }

        private void SoftMaxActivation(double expSum)
        {
            for (int i = 0; i < _outputLayerCount; i++)
            {
                OutputNeuronValues[i] = Math.Exp(OutputNeuronValues[i]) / expSum;
            }
        }

        private void ComputeDerivatives(double[] factOutputSignals)
        {
            //softmax derivative
            for (int i = 0; i < _outputLayerCount; i++)
            {
                var outputValue = OutputNeuronValues[i];
                OutputNeuronDeltas[i] = factOutputSignals[i] - outputValue;
            }

            //sigmoid derivative
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                var sum = 0d;

                for (int j = 0; j < _outputLayerCount; j++)
                {
                    sum += OutputNeuronDeltas[j] * HiddenToOutputWeights[j, i];
                }

                HiddenNeuronDeltas[i] = HiddenNeuronValues[i] * (1 - HiddenNeuronValues[i]) * sum;
            }
        }

        private void ChangeWeights()
        {
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _inputLayerCount; j++)
                {
                    InputToHiddenWeights[i, j] += _nu * HiddenNeuronDeltas[i] * InputLayerValues[j];
                }
            }
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _outputLayerCount; j++)
                {
                    HiddenToOutputWeights[j, i] += _nu * OutputNeuronDeltas[j] * HiddenNeuronValues[i];
                }
            }
        }

        private double ComputeCrossEntropy()
        {
            var result = 0d;
            var output = new double[_outputLayerCount];

            foreach (var image in InputData)
            {
                Array.Clear(output, 0, _outputLayerCount);
                AllocateInputData(image);

                output[image.Label] = 1d;

                ComputeNeuronsHiddenValue();
                ComputeNeuronsOutputValues();

                for (int i = 0; i < _outputLayerCount; i++)
                {
                    result += output[i] * Math.Log(OutputNeuronValues[i]);
                }
            }

            return -result / InputData.Count;
        }
    }
}
