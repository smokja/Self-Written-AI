using ArtificialNeuralNetwork.Model;
using System;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class Trainer
    {
        int iterations = 10000;
        double diff = 0;
        public void Train(SimpleNeuralNetwork network)
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                XORCase(network, 1, 1, 1);
                //XORCase(network, 0, 0, 1);
            }


          
        }

        private void XORCase(SimpleNeuralNetwork network, int first, int second, int expected)
        {
            network.InputLayer[0].Value = 1;
            network.InputLayer[1].Value = 0;
            var output = network.CalculateResult();
            diff = 1 - output.Value;


            network.InputLayer[0].Value = 1;
            network.InputLayer[1].Value = 1;
            output = network.CalculateResult();
            diff = 0 - output.Value;
            
            var sum = network.Output.ValueWithoutSigmoid;
            var slope = Sigmoid.Derivative(sum) * diff;


            int countOutputs = 1;
            // Should be 3 hidden weights for now
            double[] hiddenWeights = new double[network.HiddenLayer.Length];
            double[] trueValues = new double[network.HiddenLayer.Length];

            for (int i = 0; i < network.HiddenLayer.Length; i++)
            {
                var neuron = network.HiddenLayer[i];
                hiddenWeights[i] = neuron.Weights[0];
                trueValues[i] = neuron.ValueWithoutSigmoid;

                for (int y = 0; y < countOutputs; y++)
                {
                    neuron.Weights[y] += (slope / neuron.Value);
                }
            }

            double[] dividedWeights = new double[hiddenWeights.Length];
            for (int i = 0; i < hiddenWeights.Length; i++)
            {
                dividedWeights[i] = slope / hiddenWeights[i];
            }

            double[] deltaHiddenSum = new double[dividedWeights.Length];
            for (int i = 0; i < trueValues.Length; i++)
            {
                deltaHiddenSum[i] = dividedWeights[i] * trueValues[i];
            }

            double[] deltaWeights = new double[network.InputLayer.Length * deltaHiddenSum.Length];
            int counter = 0;
            for (int i = 0; i < deltaHiddenSum.Length; i++)
            {
                for (int p = 0; p < network.InputLayer.Length; p++)
                {
                    deltaWeights[counter++] = deltaHiddenSum[i] / network.InputLayer[p].Value;
                }
            }

            counter = 0;
            for (int i = 0; i < network.InputLayer.Length; i++)
            {
                for (int p = 0; p < network.InputLayer[i].Weights.Length; p++)
                {
                    network.InputLayer[i].Weights[p] += deltaWeights[counter++];
                }
            }
        }
    }
}
