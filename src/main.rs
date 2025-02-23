use burn::backend::WgpuBackend;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::{Dataset, InMemDataset};
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::af::{sigmoid, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::data::FloatTensor;
use burn::tensor::Tensor;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainState};
use rand::Rng;

#[derive(Module, Debug)]
struct LinearRegression<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LinearRegression<B> {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            linear: Linear::new(LinearConfig::new(input_size, output_size)),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

#[derive(Clone, Debug)]
struct RegressionItem {
    input: f32,
    target: f32,
}

#[derive(Clone, Debug)]
struct RegressionDataset {
    items: Vec<RegressionItem>,
}

impl Dataset<RegressionItem> for RegressionDataset {
    fn get(&self, index: usize) -> Option<RegressionItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Debug)]
struct RegressionBatch<B: Backend> {
    input: Tensor<B, 2>,
    target: Tensor<B, 2>,
}

impl<B: Backend> burn::data::dataloader::Batch<RegressionItem, RegressionBatch<B>>
for RegressionBatch<B>
{
    fn from_data(items: Vec<RegressionItem>) -> Self {
        let inputs = items.iter().map(|item| item.input).collect::<Vec<_>>();
        let targets = items.iter().map(|item| item.target).collect::<Vec<_>>();

        let inputs_tensor = Tensor::<B, 2>::from_data(FloatTensor::from_vec(
            inputs.iter().map(|&val| val).collect(),
            [items.len(), 1],
        ));

        let targets_tensor = Tensor::<B, 2>::from_data(FloatTensor::from_vec(
            targets.iter().map(|&val| val).collect(),
            [items.len(), 1],
        ));

        RegressionBatch {
            input: inputs_tensor,
            target: targets_tensor,
        }
    }
}

fn generate_data(num_samples: usize, noise_std: f32) -> Vec<RegressionItem> {
    let mut rng = rand::thread_rng();
    (0..num_samples)
        .map(|_| {
            let x = rng.gen_range(-10.0..10.0);
            let noise = rng.gen_range(-noise_std..noise_std);
            let y = 2.0 * x + 1.0 + noise;
            RegressionItem { input: x, target: y }
        })
        .collect()
}

type BackendType = WgpuBackend;

fn main() {
    let train_data = generate_data(1000, 2.0);
    let test_data = generate_data(200, 2.0);

    let train_dataset = InMemDataset::new(train_data);
    let test_dataset = InMemDataset::new(test_data);

    let batch_size = 32;
    let num_epochs = 100;
    let learning_rate = 0.01;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let train_loader = DataLoaderBuilder::new(train_dataset)
        .batch_size(batch_size)
        .num_workers(4)
        .batcher::<RegressionBatch<BackendType>>();

    let test_loader = DataLoaderBuilder::new(test_dataset)
        .batch_size(batch_size)
        .num_workers(4)
        .batcher::<RegressionBatch<BackendType>>();

    let model = LinearRegression::<BackendType>::new(1, 1);

    let optim = Adam::new(AdamConfig::new().with_lr(learning_rate), model.params());

    let learner = LearnerBuilder::new(device)
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(10)
        .build(model, optim);

    let model = learner.fit(num_epochs, train_loader, test_loader);

    println!("Training complete. Trained model: {:?}", model);

}



