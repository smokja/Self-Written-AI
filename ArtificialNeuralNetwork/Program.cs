using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ArtificialNeuralNetwork.Model;

namespace ArtificialNeuralNetwork
{
    public class Program
    {
        private static SimpleNeuralNetwork snn = new SimpleNeuralNetwork();
       public static void Main(string[] args)
        {
            snn.RandomizeWeights(); // Should only be called 1 time for initial
            var trainer = new Trainer();
            trainer.Train(snn);
            Init();
        }
        public static void Init()
        {
            snn.InputLayer[0].Value = 0;
            snn.InputLayer[1].Value = 0;
            var result = snn.CalculateResult().Value;
            Console.WriteLine("--Result for 0,0 : "+result);
            
            snn.InputLayer[0].Value = 1;
            snn.InputLayer[1].Value = 0;
            result = snn.CalculateResult().Value;
            Console.WriteLine("--Result for 1,0 : " + result);

            snn.InputLayer[0].Value = 1;
            snn.InputLayer[1].Value = 1;
            result = snn.CalculateResult().Value;
            Console.WriteLine("--Result for 1,1 : " + result);

            var info = Console.ReadKey();
            if (info.KeyChar == 'x')
            {
                return;
            }

            Init();
        }
    }
}
