using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork.Model
{
    public class Neuron
    {
        public double Bias { get; set; }
        public double[] Weights { get; set; }
        public double Value { get; set; }
        public double ValueWithoutSigmoid { get; set; }
    
    }
}
