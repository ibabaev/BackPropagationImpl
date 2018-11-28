using System;
using System.Collections.Generic;
using System.Linq;

namespace hw
{
    public class NNet
    {
        public double LearningRate { get; set; }

        private double[] _inputLayerValues;
        private double[] _hiddenNeuronValues;
        private double[] _outputNeuronValues;
        private double[] _hiddenNeuronDeltas;
        private double[] _outputNeuronDeltas;

        private double[,] _inputToHiddenWeights;
        private double[,] _hiddenToOutputWeights;

        private List<Image> _inputData;

        private int _inputLayerCount;
        private int _hiddenLayerCount;
        private int _outputLayerCount;
        private double _error;

        public NNet(List<Image> inputImages, int hiddenLayerCount, int outputLayerCount, double learningRate, double error)
        {
            LearningRate = learningRate;
            _error = error;
            _hiddenLayerCount = hiddenLayerCount;
            _outputLayerCount = outputLayerCount;

            _inputData = new List<Image>(inputImages);
            AllocateInputData(_inputData[0]);

            _hiddenNeuronValues = new double[_hiddenLayerCount];
            _outputNeuronValues = new double[_outputLayerCount];
            _hiddenNeuronDeltas = new double[_hiddenLayerCount];
            _outputNeuronDeltas = new double[_outputLayerCount];
            _inputToHiddenWeights = new double[_hiddenLayerCount, _inputLayerCount];
            _hiddenToOutputWeights = new double[_outputLayerCount, _hiddenLayerCount];
        }

        public void Train(int Epochs)
        {
            var rnd = new Random();

            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _inputLayerCount; j++)
                {
                    _inputToHiddenWeights[i, j] = rnd.NextDouble() / 100d;
                }
            }
            for (int i = 0; i < _outputLayerCount; i++)
            {
                for (int j = 0; j < _hiddenLayerCount; j++)
                {
                    _hiddenToOutputWeights[i, j] = rnd.NextDouble() / 100d;
                }
            }

            var output = new double[_outputLayerCount];

            for (int i = 0; i < Epochs; i++)
            {
                var randomizedInput = _inputData.OrderBy(x => rnd.Next()).ToList();

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

                Console.WriteLine("Epoch " + i);

                var err = ComputeCrossEntropy();

                if (err < _error)
                {
                    Console.WriteLine("TrainedSuccessfully");
                    return;
                }

                Console.WriteLine("Error: " + err);
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
                if (_outputNeuronValues[i] > max)
                {
                    max = _outputNeuronValues[i];
                    maxNum = i;
                }
            }

            return maxNum;
        }

        private void AllocateInputData(Image image)
        {
            var count = image.Data.GetLength(0);
            _inputLayerCount = count * count;

            _inputLayerValues = new double[_inputLayerCount];

            for (int i = 0; i < count; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    _inputLayerValues[i + j * count] = image.Data[i, j] / 255f;
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
                    sum += _inputLayerValues[j] * _inputToHiddenWeights[i, j];
                }

                _hiddenNeuronValues[i] = SigmoidActivation(sum);
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
                    sum += _hiddenNeuronValues[j] * _hiddenToOutputWeights[i, j];
                }

                _outputNeuronValues[i] = sum;
                expSum += Math.Exp(sum);
            }

            SoftMaxActivation(expSum);
        }

        private void SoftMaxActivation(double expSum)
        {
            for (int i = 0; i < _outputLayerCount; i++)
            {
                _outputNeuronValues[i] = Math.Exp(_outputNeuronValues[i]) / expSum;
            }
        }

        private void ComputeDerivatives(double[] factOutputSignals)
        {
            //softmax derivative
            for (int i = 0; i < _outputLayerCount; i++)
            {
                _outputNeuronDeltas[i] = factOutputSignals[i] - _outputNeuronValues[i];
            }

            //sigmoid derivative
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                var sum = 0d;

                for (int j = 0; j < _outputLayerCount; j++)
                {
                    sum += _outputNeuronDeltas[j] * _hiddenToOutputWeights[j, i];
                }

                _hiddenNeuronDeltas[i] = _hiddenNeuronValues[i] * (1 - _hiddenNeuronValues[i]) * sum;
            }
        }

        private void ChangeWeights()
        {
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _inputLayerCount; j++)
                {
                    _inputToHiddenWeights[i, j] += LearningRate * _hiddenNeuronDeltas[i] * _inputLayerValues[j];
                }
            }
            for (int i = 0; i < _hiddenLayerCount; i++)
            {
                for (int j = 0; j < _outputLayerCount; j++)
                {
                    _hiddenToOutputWeights[j, i] += LearningRate * _outputNeuronDeltas[j] * _hiddenNeuronValues[i];
                }
            }
        }

        private double ComputeCrossEntropy()
        {
            var result = 0d;
            var output = new double[_outputLayerCount];

            foreach (var image in _inputData)
            {
                Array.Clear(output, 0, _outputLayerCount);
                AllocateInputData(image);

                output[image.Label] = 1d;

                ComputeNeuronsHiddenValue();
                ComputeNeuronsOutputValues();

                for (int i = 0; i < _outputLayerCount; i++)
                {
                    result += output[i] * Math.Log(_outputNeuronValues[i]);
                }
            }

            return -result / _inputData.Count;
        }
    }
}
