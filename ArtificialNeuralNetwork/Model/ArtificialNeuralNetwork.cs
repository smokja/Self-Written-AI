using System;
using System.Collections.Generic;
using System.Linq;

namespace ArtificialNeuralNetwork.Model
{
    public class ArtificialNeuralNetwork
    {
        public List<Neuron> InputLayer { get; set; }
        public List<List<Neuron>> HiddenLayers { get; set; }
        public List<Neuron> OutputLayer { get; set; }

        public ArtificialNeuralNetwork(int inputNeurons,
                                       int output,
                                       int hiddenLayers,
                                       int[] hiddenNeurons)
        {
            if (hiddenLayers != hiddenNeurons.Length)
                throw new ArgumentException("hiddenLayers and hiddenNeurons length must match");

            InputLayer = CreateLayer(inputNeurons).ToList();
            OutputLayer = CreateLayer(output).ToList();
            HiddenLayers = new List<List<Neuron>>();

            for (int i = 0; i < hiddenLayers; i++)
            {
                HiddenLayers.Add(CreateLayer(hiddenNeurons[i]).ToList());
            }

            RandomizeWeights();
        }

        private IEnumerable<Neuron> CreateLayer(int inputNeurons)
        {
            for (int i = 0; i < inputNeurons; i++)
            {
                yield return new Neuron();
            }
        }

        public void RandomizeWeights()
        {
            var random = new Random();

            RandomizeLayer(InputLayer, (HiddenLayers.FirstOrDefault() ?? OutputLayer).Count(), random);
            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                RandomizeLayer(HiddenLayers[i], (HiddenLayers.ElementAtOrDefault(i + 1) ?? OutputLayer).Count, random);
            }
        }

        public void RandomizeLayer(List<Neuron> layer, int weightCount, Random random)
        {
            foreach (var neuron in layer)
            {
                for (int i = 0; i < weightCount; i++)
                {
                    neuron.Weights.Add(random.NextDouble());
                }
            }
        }

        public List<double> FeedForward(double[] input)
        {
            if (input.Length != InputLayer.Count)
                throw new ArgumentException("input length has to be the same as the InputLayer length");

            // calculate input activations
            for (int i = 0; i < input.Length; i++)
            {
                InputLayer[i].Value = Sigmoid.Process(input[i]);
            }

            // calculate hidden layer activations
            // i : LayerIndex in HiddenLayers
            // j : NeuronIndex in HiddenLayers[i]
            for (int i = 0; i < HiddenLayers.Count; i++)
            {
                var previousLayer = HiddenLayers.ElementAtOrDefault(i - 1) ?? InputLayer;
                for (int j = 0; j < HiddenLayers[i].Count; j++)
                {
                    HiddenLayers[i][j].Value = Sigmoid.Process(CalculateWeightedSum(previousLayer, j));
                }
            }

            // calculate output activations
            // i : NeuronIndex in OutputLayer
            var lastLayer = HiddenLayers.LastOrDefault() ?? InputLayer;
            for (int i = 0; i < OutputLayer.Count; i++)
            {
                OutputLayer[i].Value = Sigmoid.Process(CalculateWeightedSum(lastLayer, i));
            }

            // return only values of output layer
            return OutputLayer.Select(x => x.Value).ToList();
        }
        double CalculateWeightedSum(List<Neuron> lastLayer, int weightIndex)
        {
            double weightedSum = 0.0;
            foreach (var neuron in lastLayer)
            {
                var weight = neuron.Weights[weightIndex];
                weightedSum += weight * neuron.Value;
            }


            return weightedSum;
        }
    }
}
