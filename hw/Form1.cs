using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace hw
{
    public partial class Form1 : Form
    {
        private List<Image> _images;

        private int _counter = 0;

        private NNet _net;

        public Form1()
        {
            InitializeComponent();
            // Task.Factory.StartNew(Console);
            Console.WriteLine("Lets start");

            _images = new List<Image>();

            foreach (var image in MnistReader.ReadTestData())
            {
                _images.Add(image);
            }

            var images = new List<Image>();

            foreach (var image in MnistReader.ReadTrainingData())
            {
                images.Add(image);
            }

            Console.WriteLine("read, starting allocating net");

            _net = new NNet(images, 50, 10, 0.01d);

            Console.WriteLine("done");

            _net.Train(25);

            _net.Test(_images);
            //DisplayData
            //foreach (var image in MnistReader.ReadTestData())
            //{
            //    for (int i = 0; i < image.Data.GetLength(0); i++)
            //    {
            //        Console.WriteLine();
            //        for (int j = 0; j < image.Data.GetLength(0); j++)
            //            if (image.Data[i, j] == 0) Console.Write(" ");
            //            else Console.Write("0");
            //    }
            //}

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            var size = _images[_counter].Data.GetLength(0);

            var btmp = new Bitmap(size, size);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    var color = _images[_counter].Data[i, j];
                    btmp.SetPixel(j, i, Color.FromArgb(color, color, color));
                }
            }
            label2.Text = _images[_counter].Label.ToString();
            pictureBox1.Image = btmp;
            label1.Text = _net.Test(_images[_counter]).ToString();


            _counter++;
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        //private void Console()
        //{
        //    // Запускаем консоль.
        //    if (AllocConsole())
        //    {
        //        System.Console.WriteLine("Для выхода наберите exit.");
        //        while (true)
        //        {
        //            // Считываем данные.
        //            string output = System.Console.ReadLine();
        //            if (output == "exit")
        //                break;
        //            // Выводим данные в textBox
        //            Action action = () => textBox.Text += output + Environment.NewLine;
        //            if (InvokeRequired)
        //                Invoke(action);
        //            else
        //                action();
        //        }
        //        // Закрываем консоль.
        //        FreeConsole();
        //    }
        //}

        //[DllImport("kernel32.dll", SetLastError = true)]
        //[return: MarshalAs(UnmanagedType.Bool)]
        //private static extern bool AllocConsole();

        //[DllImport("kernel32.dll", SetLastError = true)]
        //[return: MarshalAs(UnmanagedType.Bool)]
        //private static extern bool FreeConsole();
    }
}
