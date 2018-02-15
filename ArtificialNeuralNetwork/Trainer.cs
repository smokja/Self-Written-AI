using ArtificialNeuralNetwork.Model;
using System;
using System.Diagnostics;
using System.Linq;

namespace ArtificialNeuralNetwork
{
    public class Trainer
    {
        int iterations = 10000;
        double error_j = 0;
        public void Train(SimpleNeuralNetwork network)
        {
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                XORCase(network);
                //XORCase(network, 0, 0, 1);
            }


          
        }

        private void XORCase(SimpleNeuralNetwork network)
        {
            Neuron output = new Neuron();

            network.InputLayer[0].Value = 1;
            network.InputLayer[1].Value = 1;
            output = network.CalculateResult();
            error_j = (0 - output.Value) * Sigmoid.Derivative(output.Value);
            network.InputLayer[0].Value = 0;
            network.InputLayer[1].Value = 0;
            output = network.CalculateResult();
            error_j = (0 - output.Value) * Sigmoid.Derivative(output.Value);
            network.InputLayer[0].Value = 1;
            network.InputLayer[1].Value = 0;
            output = network.CalculateResult();
            error_j = (1 - output.Value) * Sigmoid.Derivative(output.Value);
            network.InputLayer[0].Value = 0;
            network.InputLayer[1].Value = 1;
            output = network.CalculateResult();
            error_j = (1 - output.Value) * Sigmoid.Derivative(output.Value);
            var error = (output.Weights.ToList().Select(x => x * error_j)).Select(x => x * Sigmoid.Derivative(output.Value));
        }
    }
}
