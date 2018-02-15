using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork.Model
{
    public class SimpleNeuralNetwork
    {
        public Neuron[] InputLayer { get; set; } = new Neuron[2];
        public Neuron[] HiddenLayer { get; set; } = new Neuron[3];
        public Neuron Output { get; set; } = new Neuron();
        public SimpleNeuralNetwork()
        {
            for (int i = 0; i < InputLayer.Length; i++)
            {
                InputLayer[i] = new Neuron();
            }

            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                HiddenLayer[i] = new Neuron();
            }

        }

        public void RandomizeWeights()
        {
            var random = new Random();
            for (int i = 0; i < InputLayer.Length; i++)
            {
                InputLayer[i].Weights = new double[HiddenLayer.Length];
                for (int p = 0; p < HiddenLayer.Length; p++)
                {
                    InputLayer[i].Weights[p] = random.NextDouble();
                }
            }

            for (int i = 0; i < HiddenLayer.Length; i++)
            {
                HiddenLayer[i].Weights = new double[1];
                HiddenLayer[i].Weights[0] = random.NextDouble();
            }
        }

        public Neuron CalculateResult()
        {
            Output.ValueWithoutSigmoid = 0;
           
            for (int inputNeuronIndex = 0; inputNeuronIndex < InputLayer.Length; inputNeuronIndex++)
            {
                var inputNeuron = InputLayer[inputNeuronIndex];
                // First reset the value withoutsigmoid
                for (int weightIndex = 0; weightIndex < inputNeuron.Weights.Length; weightIndex++)
                {
                    HiddenLayer[weightIndex].ValueWithoutSigmoid = 0;
                }

                for (int weightIndex = 0; weightIndex < inputNeuron.Weights.Length; weightIndex++)
                {
                    HiddenLayer[weightIndex].ValueWithoutSigmoid += inputNeuron.Weights[weightIndex] * inputNeuron.Value;
                }
            }

            for (int hiddenNeuronIndex = 0; hiddenNeuronIndex < HiddenLayer.Length; hiddenNeuronIndex++)
            {
                var hiddenNeuron = HiddenLayer[hiddenNeuronIndex];
                hiddenNeuron.Value = Sigmoid.Process(hiddenNeuron.ValueWithoutSigmoid);
                Output.ValueWithoutSigmoid += hiddenNeuron.Weights[0] * hiddenNeuron.Value;
            }
            
            Output.Value = Sigmoid.Process(Output.ValueWithoutSigmoid);
            return Output;
        }
        
    }
}
