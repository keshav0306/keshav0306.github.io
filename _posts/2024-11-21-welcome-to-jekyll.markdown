---
layout: post
title:  "Interpreting a Linear Classifier"
date:   2024-11-21 20:26:41 +0530
categories: jekyll update
---
<!-- You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->

### Analysing a simple linear classifier

Trained using gradient descent. The dataset consists of N random vectors of a fixed dimension D. A linear transformation W of shape (N * D) transforms the D dimensional vector to N dimensional. The model is trained using cross entropy loss.

This is the dataset class used
```
class Synth2d(Dataset):

def __init__(self, num_samples):
	self.num_samples = num_samples
	self.data = torch.randn(self.num_samples, 512)

def __len__(self):
	return self.num_samples
	
def __getitem__(self, idx):
	return self.data[idx], idx
```

This is the model code which is just a linear layer
```
class Model(nn.Module):

def __init__(self):

	super().__init__()
	self.cls_proj = nn.Linear(512, 256)

def forward(self, x):
	out = self.cls_proj(x)
	return out
```

Lets analyze how the model trains and lets try to explain the loss curve at every timestep and how it evolves. This will not only help us analyze how gradient descent actually evolves the randomly initialized matrix over time, but also help us practice the approach for more complex models which might evolve this as a sub component like using an attention network for classification.

For beginning, the following optimizer is used
```
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

#### Minibatch Analysis

Lets analyze how a single minibatch affects the gradient of the weight matrix and lets see the gradient analytical form first to get to know how the weight matrix is going to change.
If for a batch containing B samples of the form $(s_i, y_i)$, the loss is going to be the cross entropy loss.
$$L_i = -y_{ij} . log(p(x_{ij})) - (1-y_{ij}) * log(1 - p(x_{ij}))$$
The logits $x_i$ will be a vector of N dimensions representing the logits of the classification model. $y_{i}$ is going to be a one hot vector with 1 at the idx of the label and zeros elsewhere.

The upstream gradient is going to be, if
$$\frac{1}{p_i} * \frac{\partial {p_i}}{\partial {x_j}} $$

where for the postion where the logit idx equals the gt label idx
$$\frac{\partial {p_i}}{\partial {x_i}} = p_i * (1 - p_i)$$

and for other places
$$\frac{\partial {p_i}}{\partial {x_j}} = p_i * p_j$$

combined we get the gradient of the loss wrt the logit $x_{ij}$
$$\frac{\partial {L_i}}{\partial {x_{ij}}} = p_j - \partial_{ij}$$

Lets see this in code
```

batch_size = 256
dataset = Synth2d()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch = 1
for _ in range(epoch):
	for idx, sample in enumerate(dataloader):
		vec, cls = sample
		out = model(vec)
		probs = F.softmax(out, -1)
		loss_batch = -torch.log(probs[torch.arange(batch_size), cls])
		loss = loss_batch.mean()
		out.retain_grad()
		probs.retain_grad()
	
		optimizer.zero_grad()	
		loss.backward()
		# optimizer.step()
	
		print(probs[0])	
		print(cls[0])
		print(out.grad[0] * 256)

```

which correctly outputs the gradient of the loss wrt the logits which for D=3, N=6 gives
```
tensor([0.1063, 0.2938, 0.1671, 0.1387, 0.0675, 0.2265], grad_fn=<SelectBackward0>) 
tensor(0)
tensor([-0.8937, 0.2938, 0.1671, 0.1387, 0.0675, 0.2265])
```

We see that the gradient of the loss wrt the logits for the first element of the first sample comes out to be negative and the largest. This is expected in the beginning as the probs should be unbiased and should be all equal to $\frac{1}{N}$ . The gradient of the first element comes out to be $\frac{1}{N} - 1$.
which is going to be negative and the others are going to be positive and equal to $\frac{1}{N}$. 

Thus the gradient is basically saying that moving the first element by the same factor $d$, a small positive direction as others would decrease the loss by $0.8937 * d$. 

This gradient vector will act as a supervision signal for the weight matrix. The weight matrix will look at this gradient and the parameters will update accordingly. Lets compute the gradient of the weights wrt a single sample first.

$$\frac{\partial {L_i}}{\partial {W}} = \sum_{j=1}^{N} \frac{\partial {L_i}}{\partial {x_{ij}}} * \frac{\partial {x_{ij}}}{\partial {W}}$$
which would become
$$\frac{\partial {L_i}}{\partial {W}} = \frac{\partial {L_i}}{\partial {x_{i}}} * s_i^{T}$$
In code this becomes 

```
print(out.grad.sum(0))
print(model.cls_proj.bias.grad)

grad_w = (out.grad[..., None] @ vec[:, None]).sum(0)

print(grad_w)
print(model.cls_proj.weight.grad)
```

The output comes out to be
```
tensor([-0.0502, 0.1028, 0.1481, -0.0695, -0.1002, -0.0311])
tensor([-0.0502, 0.1028, 0.1481, -0.0695, -0.1002, -0.0311])
tensor([[-0.1122, -0.1688, 0.0796], [ 0.0497, 0.2917, -0.0425], [ 0.0792, 0.2248, -0.1936], [-0.1343, -0.1557, 0.1022], [ 0.0147, -0.2316, 0.2726], [ 0.1028, 0.0396, -0.2183]]) 
tensor([[-0.1122, -0.1688, 0.0796], [ 0.0497, 0.2917, -0.0425], [ 0.0792, 0.2248, -0.1936], [-0.1343, -0.1557, 0.1022], [ 0.0147, -0.2316, 0.2726], [ 0.1028, 0.0396, -0.2183]])
```

Does the gradient have a meaning that we can visualize?
Lets look at the gradient of a column of W and see if it has a meaning or not that we can understand as of now. The $j^{th}$ column is going to be  $\frac{\partial {L_i}}{\partial {x_{i}}} * s_{ij}$. which means that whatever the current column is (i.e the N dim vector) after the update is going to shift in the the direction of the upstream gradient times the scaling factor governed by the input, which makes sense. For a minibatch the gradient is going to move the coloum in the average of the changes.

We are concerned with getting the intuition of how the matrix and the bias is going to update with iterations. We have computed the analytical gradient for freshing up concepts as well but now we need to analyse the updates.

It is important to visualize the entire minibatch

This is the data and we concern ourselves with full batch gradient descent for now.
```
tensor([[-2.3987, 0.2298, 1.8910],
		[ 0.2436, 1.2445, -0.4422],
		[-0.2560, 0.2610, 1.5425],
		[-1.2028, 0.2232, 0.3578],
		[-1.9753, -0.1499, -1.0105],
		[-0.5265, -0.7566, -0.4501]])
```


So, the question is can these 6 vectors be mapped to 6 perpendicular directions in the 6 dim vector space, and what is the training dynamics?