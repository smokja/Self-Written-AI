using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork.Model
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        private double _value;
        public double ActivationValue { 
            get 
            { 
                return _value;
            } 
            set
            {
                _value = Sigmoid.Process(Bias + value);
            } 
        }
        public double Bias { get;set; }
        public Neuron()
        {
            Weights = new List<double>();
        }
    }
}
