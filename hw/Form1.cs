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

            _images = new List<Image>();
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

        private void button2_Click(object sender, EventArgs e)
        {
            var hidden = int.Parse(textBox1.Text);
            var output = int.Parse(textBox2.Text);
            var lrate = double.Parse(textBox3.Text);
            var error = double.Parse(textBox4.Text);
            var epochs = int.Parse(textBox5.Text);

            foreach (var image in MnistReader.ReadTestData())
            {
                _images.Add(image);
            }

            var images = new List<Image>();

            foreach (var image in MnistReader.ReadTrainingData())
            {
                images.Add(image);
            }

            Console.WriteLine("Loading Mnist Data: Done");
            Console.WriteLine(hidden + "||" + output + "||" + lrate + "||" + error);
           _net = new NNet(images, hidden, output, lrate, error);

            _net.Train(epochs);

            _net.Test(_images);
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox4_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox5_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
