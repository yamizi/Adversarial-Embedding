#from https://github.com/brijeshiitg/Pytorch-Implementation-of-YeNet-Deep-Learning-Hierarchical-Representations-for-Image-Steganalysis-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import traceback

# print(srm_filters.shape)

def trainYenet(train_x,train_y,Lr = 1e-5, batch_size = 5, epochs=5, test_x=None, test_y=None, experiment=None):
	set_size = len(train_x)
	model = Yenet().cuda().train()

	r = torch.randperm(set_size)
	train_x = train_x.cuda()
	train_y = train_y.cuda()

	xp_epochs = experiment.params.get("detector_epochs",None)
	epochs = epochs if xp_epochs is None else xp_epochs

	criterion = torch.nn.CrossEntropyLoss() #torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=Lr)

	for i in range(0, epochs * set_size, batch_size):
		input_tensor = train_x[i % set_size: i % set_size + batch_size].float().cuda()
		output_tensor = train_y[i % set_size: i % set_size + batch_size].long().cuda()

		predicted_value = model(input_tensor)


		loss = criterion(predicted_value, output_tensor)

		print(i,loss)
		if experiment is not None:
			experiment.log_metric("detector_loss",loss,step=i)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()


	if test_x is not None:
		model.eval()
		with torch.no_grad():
			from sklearn.metrics import roc_auc_score, accuracy_score
			predictions_proba = model(test_x.cuda())
			predictions = torch.argmax(predictions_proba,1).cpu().detach().numpy()
			true = test_y.cpu().detach().numpy()

			acc = accuracy_score(true,predictions)
			print("accuracy {}".format(acc))

			auc = 0
			try:
				auc = roc_auc_score(true,predictions)
				print("auc {}".format(auc))
			except Exception as e:
				print(e)
				traceback.print_exc()


		if experiment is not None:
			experiment.log_metric("detector_accuracy",acc)
			experiment.log_metric("detector_auc", auc)

	return model

class Yenet(nn.Module):
	def __init__(self):
		super(Yenet, self).__init__()

		self.tlu = nn.Hardtanh(min_val=-3.0, max_val=3.0)

		self.conv2 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

		self.conv3 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

		self.conv4 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=0)

		self.conv5 = nn.Conv2d(30, 32, kernel_size=5, stride=1, padding=0)
		
		self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)

		self.conv7 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)

		self.conv8 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)

		self.conv9 = nn.Conv2d(16, 16, kernel_size=3, stride=3, padding=0)

		self.fc = nn.Linear(16*3*3, 2)

		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		srm_16filters = np.load('./exploration/srm_16_filters.npy')
		srm_minmax = np.load('./exploration/minmax_filters.npy')
		srm_filters = np.concatenate((srm_16filters, srm_minmax), axis=0)

		srm_filters = torch.from_numpy(srm_filters).to(device=device, dtype=torch.float)
		self.srm_filters = torch.autograd.Variable(srm_filters, requires_grad=True)

	def forward(self, x):
		x = T.Resize(256)(x)
		x = T.Grayscale(num_output_channels=1)(x)
		out = self.tlu(F.conv2d(x, self.srm_filters))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = F.relu(self.conv4(out))
		out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=1)
		out = F.relu(self.conv5(out))
		out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
		out = F.relu(self.conv6(out))
		out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
		out = F.relu(self.conv7(out))
		out = F.avg_pool2d(out, kernel_size=3, stride=2, padding=0)
		out = F.relu(self.conv8(out))
		out = F.relu(self.conv9(out))
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out
