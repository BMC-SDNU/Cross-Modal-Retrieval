# 2021-CVPR-MRL
Peng Hu, Xi Peng, Hongyuan Zhu, Liangli Zhen, Jie Lin, [Learning Cross-Modal Retrieval with Noisy Labels](paper/Learning_Cross_Modal_Retrieval_with_Noisy_Labels.pdf), IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 19-25, 2021. (PyTorch Code)

## Abstract
Recently, cross-modal retrieval is emerging with the help of deep multimodal learning. However, even for unimodal data, collecting large-scale well-annotated data is expensive and time-consuming, and not to mention the additional challenges from multiple modalities. Although crowd-sourcing annotation, *e.g.*, Amazon's Mechanical Turk, can be utilized to mitigate the labeling cost, but leading to the unavoidable noise in labels for the non-expert annotating. To tackle the challenge, this paper presents a general Multimodal Robust Learning framework (MRL) for learning with multimodal noisy labels to mitigate noisy samples and correlate distinct modalities simultaneously. To be specific, we propose a Robust Clustering loss (RC) to make the deep networks focus on clean samples instead of noisy ones. Besides, a simple yet effective multimodal loss function, called Multimodal Contrastive loss (MC), is proposed to maximize the mutual information between different modalities, thus alleviating the interference of noisy samples and cross-modal discrepancy. Extensive experiments are conducted on four widely-used multimodal datasets to demonstrate the effectiveness of the proposed approach by comparing to 14 state-of-the-art methods.

## Framework
<h4>Figure 1 The pipeline of the proposed method for 𝓂 modalities, <i>e.g.</i>, images 𝒳₁ with noisy labels 𝒴₁, and texts 𝒳<sub>𝓂</sub> with noisy labels 𝒴<sub>𝓂</sub>. The modality-specific networks learn common representations for 𝓂 different modalities. The Robust Clustering loss &Lscr;<sub>𝓇</sub> is adopted to mitigate the noise in labels for learning discrimination and narrow the heterogeneous gap. The outputs of networks interact with each other to learn common representations by using instance- and pair-level contrast, <i>i.e.</i>, multimodal contrastive learning (&Lscr;<sub>𝒸</sub>), thus further mitigating noisy labels and cross-modal discrepancy. &Lscr;<sub>𝒸</sub> tries to maximally scatter inter-modal samples while compacting intra-modal points over the common unit sphere/space.</h4>

![MRL](paper/MRL_Framework.jpg)

## Usage
To train a model with 0.6 noise rate on Wikipedia, just run main_noisy.py:
```bash
python main_noisy.py --max_epochs 30 --log_name noisylabel_mce --loss MCE  --lr 0.0001 --train_batch_size 100 --beta 0.7 --noisy_ratio 0.6 --data_name wiki
```
You can get outputs as follows:
```
Epoch: 24 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 277ms | Loss: 2.365 | LR: 1.28428e-05
 
Validation: Img2Txt: 0.480904	Txt2Img: 0.436563	Avg: 0.458733
Test: Img2Txt: 0.474708	Txt2Img: 0.440001	Avg: 0.457354
Saving..

Epoch: 25 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 275ms | Loss: 2.362 | LR: 9.54915e-06
 
Validation: Img2Txt: 0.48379	Txt2Img: 0.437549	Avg: 0.460669
Test: Img2Txt: 0.475301	Txt2Img: 0.44056	Avg: 0.45793
Saving..

Epoch: 26 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 276ms | Loss: 2.361 | LR: 6.69873e-06
 
Validation: Img2Txt: 0.482946	Txt2Img: 0.43729	Avg: 0.460118

Epoch: 27 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 273ms | Loss: 2.360 | LR: 4.32273e-06
 
Validation: Img2Txt: 0.480506	Txt2Img: 0.437512	Avg: 0.459009

Epoch: 28 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 269ms | Loss: 2.360 | LR: 2.44717e-06
 
Validation: Img2Txt: 0.481429	Txt2Img: 0.437096	Avg: 0.459263

Epoch: 29 / 30
 [================= 22/22 ==================>..]  Step: 12ms | Tot: 275ms | Loss: 2.359 | LR: 1.09262e-06
 
Validation: Img2Txt: 0.482126	Txt2Img: 0.437257	Avg: 0.459691
Evaluation on Last Epoch:
Img2Txt: 0.475	Txt2Img: 0.440	
Evaluation on Best Validation:
Img2Txt: 0.475	Txt2Img: 0.441
```

## Comparison with the State-of-the-Art
<table>
<thead>
  <h4>Table 1: Performance comparison in terms of MAP scores under the symmetric noise rates of 0.2, 0.4, 0.6 and 0.8 on the Wikipedia and INRIA-Websearch datasets. The highest MAP score is shown in <b>bold</b>.</h4>
  <tr>
    <th class="tg-0pky" rowspan="3", align="center">Method</th>
    <th class="tg-c3ow" colspan="8", align="center">Wikipedia</th>
    <th class="tg-c3ow" colspan="8", align="center">INRIA-Websearch</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4", align="center">Image → Text</td>
    <td class="tg-c3ow" colspan="4", align="center">Text → Image</td>
    <td class="tg-c3ow" colspan="4", align="center">Image → Text</td>
    <td class="tg-c3ow" colspan="4", align="center">Text → Image</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">MCCA</td>
    <td class="tg-c3ow">0.202</td>
    <td class="tg-c3ow">0.202</td>
    <td class="tg-c3ow">0.202</td>
    <td class="tg-c3ow">0.202</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.275</td>
    <td class="tg-c3ow">0.275</td>
    <td class="tg-c3ow">0.275</td>
    <td class="tg-c3ow">0.275</td>
    <td class="tg-c3ow">0.277</td>
    <td class="tg-c3ow">0.277</td>
    <td class="tg-c3ow">0.277</td>
    <td class="tg-c3ow">0.277</td>
  </tr>
  <tr>
    <td class="tg-0pky">PLS</td>
    <td class="tg-c3ow">0.337</td>
    <td class="tg-c3ow">0.337</td>
    <td class="tg-c3ow">0.337</td>
    <td class="tg-c3ow">0.337</td>
    <td class="tg-c3ow">0.320</td>
    <td class="tg-c3ow">0.320</td>
    <td class="tg-c3ow">0.320</td>
    <td class="tg-c3ow">0.320</td>
    <td class="tg-c3ow">0.387</td>
    <td class="tg-c3ow">0.387</td>
    <td class="tg-c3ow">0.387</td>
    <td class="tg-c3ow">0.387</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.398</td>
  </tr>
  <tr>
    <td class="tg-0pky">DCCA</td>
    <td class="tg-c3ow">0.281</td>
    <td class="tg-c3ow">0.281</td>
    <td class="tg-c3ow">0.281</td>
    <td class="tg-c3ow">0.281</td>
    <td class="tg-c3ow">0.260</td>
    <td class="tg-c3ow">0.260</td>
    <td class="tg-c3ow">0.260</td>
    <td class="tg-c3ow">0.260</td>
    <td class="tg-c3ow">0.188</td>
    <td class="tg-c3ow">0.188</td>
    <td class="tg-c3ow">0.188</td>
    <td class="tg-c3ow">0.188</td>
    <td class="tg-c3ow">0.182</td>
    <td class="tg-c3ow">0.182</td>
    <td class="tg-c3ow">0.182</td>
    <td class="tg-c3ow">0.182</td>
  </tr>
  <tr>
    <td class="tg-0pky">DCCAE</td>
    <td class="tg-c3ow">0.308</td>
    <td class="tg-c3ow">0.308</td>
    <td class="tg-c3ow">0.308</td>
    <td class="tg-c3ow">0.308</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.167</td>
    <td class="tg-c3ow">0.167</td>
    <td class="tg-c3ow">0.167</td>
    <td class="tg-c3ow">0.167</td>
    <td class="tg-c3ow">0.164</td>
    <td class="tg-c3ow">0.164</td>
    <td class="tg-c3ow">0.164</td>
    <td class="tg-c3ow">0.164</td>
  </tr>
  <tr>
    <td class="tg-0pky">GMA</td>
    <td class="tg-c3ow">0.200</td>
    <td class="tg-c3ow">0.178</td>
    <td class="tg-c3ow">0.153</td>
    <td class="tg-c3ow">0.139</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.160</td>
    <td class="tg-c3ow">0.141</td>
    <td class="tg-c3ow">0.136</td>
    <td class="tg-c3ow">0.425</td>
    <td class="tg-c3ow">0.372</td>
    <td class="tg-c3ow">0.303</td>
    <td class="tg-c3ow">0.245</td>
    <td class="tg-c3ow">0.437</td>
    <td class="tg-c3ow">0.378</td>
    <td class="tg-c3ow">0.315</td>
    <td class="tg-c3ow">0.251</td>
  </tr>
  <tr>
    <td class="tg-0pky">MvDA</td>
    <td class="tg-c3ow">0.379</td>
    <td class="tg-c3ow">0.285</td>
    <td class="tg-c3ow">0.217</td>
    <td class="tg-c3ow">0.144</td>
    <td class="tg-c3ow">0.350</td>
    <td class="tg-c3ow">0.270</td>
    <td class="tg-c3ow">0.207</td>
    <td class="tg-c3ow">0.142</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.269</td>
    <td class="tg-c3ow">0.234</td>
    <td class="tg-c3ow">0.186</td>
    <td class="tg-c3ow">0.285</td>
    <td class="tg-c3ow">0.265</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.185</td>
  </tr>
  <tr>
    <td class="tg-0pky">MvDA-VC</td>
    <td class="tg-c3ow">0.389</td>
    <td class="tg-c3ow">0.330</td>
    <td class="tg-c3ow">0.256</td>
    <td class="tg-c3ow">0.162</td>
    <td class="tg-c3ow">0.355</td>
    <td class="tg-c3ow">0.304</td>
    <td class="tg-c3ow">0.241</td>
    <td class="tg-c3ow">0.153</td>
    <td class="tg-c3ow">0.288</td>
    <td class="tg-c3ow">0.272</td>
    <td class="tg-c3ow">0.241</td>
    <td class="tg-c3ow">0.192</td>
    <td class="tg-c3ow">0.286</td>
    <td class="tg-c3ow">0.268</td>
    <td class="tg-c3ow">0.238</td>
    <td class="tg-c3ow">0.190</td>
  </tr>
  <tr>
    <td class="tg-0pky">GSS-SL</td>
    <td class="tg-c3ow">0.444</td>
    <td class="tg-c3ow">0.390</td>
    <td class="tg-c3ow">0.309</td>
    <td class="tg-c3ow">0.174</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.353</td>
    <td class="tg-c3ow">0.287</td>
    <td class="tg-c3ow">0.169</td>
    <td class="tg-c3ow">0.487</td>
    <td class="tg-c3ow">0.424</td>
    <td class="tg-c3ow">0.272</td>
    <td class="tg-c3ow">0.075</td>
    <td class="tg-c3ow">0.510</td>
    <td class="tg-c3ow">0.451</td>
    <td class="tg-c3ow">0.307</td>
    <td class="tg-c3ow">0.085</td>
  </tr>
  <tr>
    <td class="tg-0pky">ACMR</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.231</td>
    <td class="tg-c3ow">0.198</td>
    <td class="tg-c3ow">0.135</td>
    <td class="tg-c3ow">0.285</td>
    <td class="tg-c3ow">0.194</td>
    <td class="tg-c3ow">0.183</td>
    <td class="tg-c3ow">0.138</td>
    <td class="tg-c3ow">0.175</td>
    <td class="tg-c3ow">0.096</td>
    <td class="tg-c3ow">0.055</td>
    <td class="tg-c3ow">0.023</td>
    <td class="tg-c3ow">0.157</td>
    <td class="tg-c3ow">0.114</td>
    <td class="tg-c3ow">0.048</td>
    <td class="tg-c3ow">0.021</td>
  </tr>
  <tr>
    <td class="tg-0pky">deep-SM</td>
    <td class="tg-c3ow">0.441</td>
    <td class="tg-c3ow">0.387</td>
    <td class="tg-c3ow">0.293</td>
    <td class="tg-c3ow">0.178</td>
    <td class="tg-c3ow">0.392</td>
    <td class="tg-c3ow">0.364</td>
    <td class="tg-c3ow">0.248</td>
    <td class="tg-c3ow">0.177</td>
    <td class="tg-c3ow">0.495</td>
    <td class="tg-c3ow">0.422</td>
    <td class="tg-c3ow">0.238</td>
    <td class="tg-c3ow">0.046</td>
    <td class="tg-c3ow">0.509</td>
    <td class="tg-c3ow">0.421</td>
    <td class="tg-c3ow">0.258</td>
    <td class="tg-c3ow">0.063</td>
  </tr>
  <tr>
    <td class="tg-0pky">FGCrossNet</td>
    <td class="tg-c3ow">0.403</td>
    <td class="tg-c3ow">0.322</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.156</td>
    <td class="tg-c3ow">0.358</td>
    <td class="tg-c3ow">0.284</td>
    <td class="tg-c3ow">0.205</td>
    <td class="tg-c3ow">0.147</td>
    <td class="tg-c3ow">0.278</td>
    <td class="tg-c3ow">0.192</td>
    <td class="tg-c3ow">0.105</td>
    <td class="tg-c3ow">0.027</td>
    <td class="tg-c3ow">0.261</td>
    <td class="tg-c3ow">0.189</td>
    <td class="tg-c3ow">0.096</td>
    <td class="tg-c3ow">0.025</td>
  </tr>
  <tr>
    <td class="tg-0pky">SDML</td>
    <td class="tg-c3ow">0.464</td>
    <td class="tg-c3ow">0.406</td>
    <td class="tg-c3ow">0.299</td>
    <td class="tg-c3ow">0.170</td>
    <td class="tg-c3ow">0.448</td>
    <td class="tg-c3ow">0.398</td>
    <td class="tg-c3ow">0.311</td>
    <td class="tg-c3ow">0.184</td>
    <td class="tg-c3ow">0.506</td>
    <td class="tg-c3ow">0.419</td>
    <td class="tg-c3ow">0.283</td>
    <td class="tg-c3ow">0.024</td>
    <td class="tg-c3ow">0.512</td>
    <td class="tg-c3ow">0.412</td>
    <td class="tg-c3ow">0.241</td>
    <td class="tg-c3ow">0.066</td>
  </tr>
  <tr>
    <td class="tg-0pky">DSCMR</td>
    <td class="tg-c3ow">0.426</td>
    <td class="tg-c3ow">0.331</td>
    <td class="tg-c3ow">0.226</td>
    <td class="tg-c3ow">0.142</td>
    <td class="tg-c3ow">0.390</td>
    <td class="tg-c3ow">0.300</td>
    <td class="tg-c3ow">0.212</td>
    <td class="tg-c3ow">0.140</td>
    <td class="tg-c3ow">0.500</td>
    <td class="tg-c3ow">0.413</td>
    <td class="tg-c3ow">0.225</td>
    <td class="tg-c3ow">0.055</td>
    <td class="tg-c3ow">0.536</td>
    <td class="tg-c3ow">0.464</td>
    <td class="tg-c3ow">0.237</td>
    <td class="tg-c3ow">0.052</td>
  </tr>
  <tr>
    <td class="tg-0pky">SMLN</td>
    <td class="tg-c3ow">0.449</td>
    <td class="tg-c3ow">0.365</td>
    <td class="tg-c3ow">0.275</td>
    <td class="tg-c3ow">0.251</td>
    <td class="tg-c3ow">0.403</td>
    <td class="tg-c3ow">0.319</td>
    <td class="tg-c3ow">0.246</td>
    <td class="tg-c3ow">0.237</td>
    <td class="tg-c3ow">0.331</td>
    <td class="tg-c3ow">0.291</td>
    <td class="tg-c3ow">0.262</td>
    <td class="tg-c3ow">0.214</td>
    <td class="tg-c3ow">0.391</td>
    <td class="tg-c3ow">0.349</td>
    <td class="tg-c3ow">0.292</td>
    <td class="tg-c3ow">0.254</td>
  </tr>
  <tr>
    <b><td class="tg-0pky">Ours</td></b>
    <td class="tg-c3ow"><b>0.514</b></td>
    <td class="tg-7btt"><b>0.491</b></td>
    <td class="tg-7btt"><b>0.464</b></td>
    <td class="tg-7btt"><b>0.435</b></td>
    <td class="tg-7btt"><b>0.461</b></td>
    <td class="tg-7btt"><b>0.453</b></td>
    <td class="tg-7btt"><b>0.421</b></td>
    <td class="tg-7btt"><b>0.400</b></td>
    <td class="tg-7btt"><b>0.559</b></td>
    <td class="tg-7btt"><b>0.543</b></td>
    <td class="tg-7btt"><b>0.512</b></td>
    <td class="tg-7btt"><b>0.417</b></td>
    <td class="tg-7btt"><b>0.587</b></td>
    <td class="tg-7btt"><b>0.571</b></td>
    <td class="tg-7btt"><b>0.533</b></td>
    <td class="tg-7btt"><b>0.424</b></td>
  </tr>
</tbody>
</table>


<table>
<thead>
  <h4>Table 2: Performance comparison in terms of MAP scores under the symmetric noise rates of 0.2, 0.4, 0.6 and 0.8 on the NUS-WIDE and XMediaNet datasets. The highest MAP score is shown in <b>bold</b>.</h4>
  <tr>
    <th class="tg-0pky" rowspan="3">Method</th>
    <th class="tg-c3ow" colspan="8">NUS-WIDE</th>
    <th class="tg-c3ow" colspan="8">XMediaNet</th>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="4", align="center">Image → Text</td>
    <td class="tg-c3ow" colspan="4", align="center">Text → Image</td>
    <td class="tg-c3ow" colspan="4", align="center">Image → Text</td>
    <td class="tg-c3ow" colspan="4", align="center">Text → Image</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-c3ow">0.4</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-c3ow">0.8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">MCCA</td>
    <td class="tg-c3ow">0.523</td>
    <td class="tg-c3ow">0.523</td>
    <td class="tg-c3ow">0.523</td>
    <td class="tg-c3ow">0.523</td>
    <td class="tg-c3ow">0.539</td>
    <td class="tg-c3ow">0.539</td>
    <td class="tg-c3ow">0.539</td>
    <td class="tg-c3ow">0.539</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.233</td>
    <td class="tg-c3ow">0.249</td>
    <td class="tg-c3ow">0.249</td>
    <td class="tg-c3ow">0.249</td>
    <td class="tg-c3ow">0.249</td>
  </tr>
  <tr>
    <td class="tg-0pky">PLS</td>
    <td class="tg-c3ow">0.498</td>
    <td class="tg-c3ow">0.498</td>
    <td class="tg-c3ow">0.498</td>
    <td class="tg-c3ow">0.498</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.266</td>
    <td class="tg-c3ow">0.266</td>
    <td class="tg-c3ow">0.266</td>
    <td class="tg-c3ow">0.266</td>
  </tr>
  <tr>
    <td class="tg-0pky">DCCA</td>
    <td class="tg-c3ow">0.527</td>
    <td class="tg-c3ow">0.527</td>
    <td class="tg-c3ow">0.527</td>
    <td class="tg-c3ow">0.527</td>
    <td class="tg-c3ow">0.537</td>
    <td class="tg-c3ow">0.537</td>
    <td class="tg-c3ow">0.537</td>
    <td class="tg-c3ow">0.537</td>
    <td class="tg-c3ow">0.152</td>
    <td class="tg-c3ow">0.152</td>
    <td class="tg-c3ow">0.152</td>
    <td class="tg-c3ow">0.152</td>
    <td class="tg-c3ow">0.162</td>
    <td class="tg-c3ow">0.162</td>
    <td class="tg-c3ow">0.162</td>
    <td class="tg-c3ow">0.162</td>
  </tr>
  <tr>
    <td class="tg-0pky">DCCAE</td>
    <td class="tg-c3ow">0.529</td>
    <td class="tg-c3ow">0.529</td>
    <td class="tg-c3ow">0.529</td>
    <td class="tg-c3ow">0.529</td>
    <td class="tg-c3ow">0.538</td>
    <td class="tg-c3ow">0.538</td>
    <td class="tg-c3ow">0.538</td>
    <td class="tg-c3ow">0.538</td>
    <td class="tg-c3ow">0.149</td>
    <td class="tg-c3ow">0.149</td>
    <td class="tg-c3ow">0.149</td>
    <td class="tg-c3ow">0.149</td>
    <td class="tg-c3ow">0.159</td>
    <td class="tg-c3ow">0.159</td>
    <td class="tg-c3ow">0.159</td>
    <td class="tg-c3ow">0.159</td>
  </tr>
  <tr>
    <td class="tg-0pky">GMA</td>
    <td class="tg-c3ow">0.545</td>
    <td class="tg-c3ow">0.515</td>
    <td class="tg-c3ow">0.488</td>
    <td class="tg-c3ow">0.469</td>
    <td class="tg-c3ow">0.547</td>
    <td class="tg-c3ow">0.517</td>
    <td class="tg-c3ow">0.491</td>
    <td class="tg-c3ow">0.475</td>
    <td class="tg-c3ow">0.400</td>
    <td class="tg-c3ow">0.380</td>
    <td class="tg-c3ow">0.344</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.376</td>
    <td class="tg-c3ow">0.364</td>
    <td class="tg-c3ow">0.336</td>
    <td class="tg-c3ow">0.277</td>
  </tr>
  <tr>
    <td class="tg-0pky">MvDA</td>
    <td class="tg-c3ow">0.590</td>
    <td class="tg-c3ow">0.551</td>
    <td class="tg-c3ow">0.568</td>
    <td class="tg-c3ow">0.471</td>
    <td class="tg-c3ow">0.609</td>
    <td class="tg-c3ow">0.585</td>
    <td class="tg-c3ow">0.596</td>
    <td class="tg-c3ow">0.498</td>
    <td class="tg-c3ow">0.329</td>
    <td class="tg-c3ow">0.318</td>
    <td class="tg-c3ow">0.301</td>
    <td class="tg-c3ow">0.256</td>
    <td class="tg-c3ow">0.324</td>
    <td class="tg-c3ow">0.314</td>
    <td class="tg-c3ow">0.296</td>
    <td class="tg-c3ow">0.254</td>
  </tr>
  <tr>
    <td class="tg-0pky">MvDA-VC</td>
    <td class="tg-c3ow">0.531</td>
    <td class="tg-c3ow">0.491</td>
    <td class="tg-c3ow">0.512</td>
    <td class="tg-c3ow">0.421</td>
    <td class="tg-c3ow">0.567</td>
    <td class="tg-c3ow">0.525</td>
    <td class="tg-c3ow">0.550</td>
    <td class="tg-c3ow">0.434</td>
    <td class="tg-c3ow">0.331</td>
    <td class="tg-c3ow">0.319</td>
    <td class="tg-c3ow">0.306</td>
    <td class="tg-c3ow">0.274</td>
    <td class="tg-c3ow">0.322</td>
    <td class="tg-c3ow">0.310</td>
    <td class="tg-c3ow">0.296</td>
    <td class="tg-c3ow">0.265</td>
  </tr>
  <tr>
    <td class="tg-0pky">GSS-SL</td>
    <td class="tg-c3ow">0.639</td>
    <td class="tg-c3ow">0.639</td>
    <td class="tg-c3ow">0.631</td>
    <td class="tg-c3ow">0.567</td>
    <td class="tg-c3ow">0.659</td>
    <td class="tg-c3ow">0.658</td>
    <td class="tg-c3ow">0.650</td>
    <td class="tg-c3ow">0.592</td>
    <td class="tg-c3ow">0.431</td>
    <td class="tg-c3ow">0.381</td>
    <td class="tg-c3ow">0.256</td>
    <td class="tg-c3ow">0.044</td>
    <td class="tg-c3ow">0.417</td>
    <td class="tg-c3ow">0.361</td>
    <td class="tg-c3ow">0.221</td>
    <td class="tg-c3ow">0.031</td>
  </tr>
  <tr>
    <td class="tg-0pky">ACMR</td>
    <td class="tg-c3ow">0.530</td>
    <td class="tg-c3ow">0.433</td>
    <td class="tg-c3ow">0.318</td>
    <td class="tg-c3ow">0.269</td>
    <td class="tg-c3ow">0.547</td>
    <td class="tg-c3ow">0.476</td>
    <td class="tg-c3ow">0.304</td>
    <td class="tg-c3ow">0.241</td>
    <td class="tg-c3ow">0.181</td>
    <td class="tg-c3ow">0.069</td>
    <td class="tg-c3ow">0.018</td>
    <td class="tg-c3ow">0.010</td>
    <td class="tg-c3ow">0.191</td>
    <td class="tg-c3ow">0.043</td>
    <td class="tg-c3ow">0.012</td>
    <td class="tg-c3ow">0.009</td>
  </tr>
  <tr>
    <td class="tg-0pky">deep-SM</td>
    <td class="tg-c3ow">0.693</td>
    <td class="tg-c3ow">0.680</td>
    <td class="tg-c3ow">0.673</td>
    <td class="tg-c3ow">0.628</td>
    <td class="tg-c3ow">0.690</td>
    <td class="tg-c3ow">0.681</td>
    <td class="tg-c3ow">0.669</td>
    <td class="tg-c3ow">0.629</td>
    <td class="tg-c3ow">0.557</td>
    <td class="tg-c3ow">0.314</td>
    <td class="tg-c3ow">0.276</td>
    <td class="tg-c3ow">0.062</td>
    <td class="tg-c3ow">0.495</td>
    <td class="tg-c3ow">0.344</td>
    <td class="tg-c3ow">0.021</td>
    <td class="tg-c3ow">0.014</td>
  </tr>
  <tr>
    <td class="tg-0pky">FGCrossNet</td>
    <td class="tg-c3ow">0.661</td>
    <td class="tg-c3ow">0.641</td>
    <td class="tg-c3ow">0.638</td>
    <td class="tg-c3ow">0.594</td>
    <td class="tg-c3ow">0.669</td>
    <td class="tg-c3ow">0.669</td>
    <td class="tg-c3ow">0.636</td>
    <td class="tg-c3ow">0.596</td>
    <td class="tg-c3ow">0.372</td>
    <td class="tg-c3ow">0.280</td>
    <td class="tg-c3ow">0.147</td>
    <td class="tg-c3ow">0.053</td>
    <td class="tg-c3ow">0.375</td>
    <td class="tg-c3ow">0.281</td>
    <td class="tg-c3ow">0.160</td>
    <td class="tg-c3ow">0.052</td>
  </tr>
  <tr>
    <td class="tg-0pky">SDML</td>
    <td class="tg-c3ow">0.694</td>
    <td class="tg-c3ow">0.677</td>
    <td class="tg-c3ow">0.633</td>
    <td class="tg-c3ow">0.389</td>
    <td class="tg-c3ow">0.693</td>
    <td class="tg-c3ow">0.681</td>
    <td class="tg-c3ow">0.644</td>
    <td class="tg-c3ow">0.416</td>
    <td class="tg-c3ow">0.534</td>
    <td class="tg-c3ow">0.420</td>
    <td class="tg-c3ow">0.216</td>
    <td class="tg-c3ow">0.009</td>
    <td class="tg-c3ow">0.563</td>
    <td class="tg-c3ow">0.445</td>
    <td class="tg-c3ow">0.237</td>
    <td class="tg-c3ow">0.011</td>
  </tr>
  <tr>
    <td class="tg-0pky">DSCMR</td>
    <td class="tg-c3ow">0.665</td>
    <td class="tg-c3ow">0.661</td>
    <td class="tg-c3ow">0.653</td>
    <td class="tg-c3ow">0.509</td>
    <td class="tg-c3ow">0.667</td>
    <td class="tg-c3ow">0.665</td>
    <td class="tg-c3ow">0.655</td>
    <td class="tg-c3ow">0.505</td>
    <td class="tg-c3ow">0.461</td>
    <td class="tg-c3ow">0.224</td>
    <td class="tg-c3ow">0.040</td>
    <td class="tg-c3ow">0.008</td>
    <td class="tg-c3ow">0.477</td>
    <td class="tg-c3ow">0.224</td>
    <td class="tg-c3ow">0.028</td>
    <td class="tg-c3ow">0.010</td>
  </tr>
  <tr>
    <td class="tg-0pky">SMLN</td>
    <td class="tg-c3ow">0.676</td>
    <td class="tg-c3ow">0.651</td>
    <td class="tg-c3ow">0.646</td>
    <td class="tg-c3ow">0.525</td>
    <td class="tg-c3ow">0.685</td>
    <td class="tg-c3ow">0.650</td>
    <td class="tg-c3ow">0.639</td>
    <td class="tg-c3ow">0.520</td>
    <td class="tg-c3ow">0.520</td>
    <td class="tg-c3ow">0.445</td>
    <td class="tg-c3ow">0.070</td>
    <td class="tg-c3ow">0.070</td>
    <td class="tg-c3ow">0.514</td>
    <td class="tg-c3ow">0.300</td>
    <td class="tg-c3ow">0.303</td>
    <td class="tg-c3ow">0.226</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ours</td>
    <td class="tg-7btt"><b>0.696</b></td>
    <td class="tg-7btt"><b>0.690</b></td>
    <td class="tg-7btt"><b>0.686</b></td>
    <td class="tg-7btt"><b>0.669</b></td>
    <td class="tg-7btt"><b>0.697</b></td>
    <td class="tg-7btt"><b>0.695</b></td>
    <td class="tg-7btt"><b>0.688</b></td>
    <td class="tg-7btt"><b>0.673</b></td>
    <td class="tg-7btt"><b>0.625</b></td>
    <td class="tg-7btt"><b>0.581</b></td>
    <td class="tg-7btt"><b>0.384</b></td>
    <td class="tg-7btt"><b>0.334</b></td>
    <td class="tg-7btt"><b>0.623</b></td>
    <td class="tg-7btt"><b><b>0.587</b></td>
    <td class="tg-7btt"><b>0.408</b></td>
    <td class="tg-7btt"><b>0.359</b></td>
  </tr>
</tbody>
</table>

## Ablation Study
<table class="tg", align="center">
<thead>
  <h4>Table 3: Comparison between our MRL (full version) and its three counterparts (CE and two variations of MRL) under the symmetric noise rates of 0.2, 0.4, 0.6 and 0.8 on the Wikipedia dataset. The highest score is shown in <b>bold</b>.</h4>
  <tr>
    <th class="tg-0lax" rowspan="2">Method</th>
    <th class="tg-baqh" colspan="4", align="center">Image → Text</th>
  </tr>
  <tr>
    <td class="tg-0lax">0.2</td>
    <td class="tg-0lax">0.4</td>
    <td class="tg-0lax">0.6</td>
    <td class="tg-0lax">0.8</td>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">CE</td>
    <td class="tg-0lax">0.441</td>
    <td class="tg-0lax">0.387</td>
    <td class="tg-0lax">0.293</td>
    <td class="tg-0lax">0.178</td>
  </tr>
  <tr>
    <td class="tg-0lax">MRL (with &Lscr;<sub>𝓇</sub> only)</td>
    <td class="tg-0lax">0.482</td>
    <td class="tg-0lax">0.434</td>
    <td class="tg-0lax">0.363</td>
    <td class="tg-0lax">0.239</td>
  </tr>
  <tr>
    <td class="tg-0lax">MRL (with &Lscr;<sub>𝒸</sub> only)</td>
    <td class="tg-0lax">0.412</td>
    <td class="tg-0lax">0.412</td>
    <td class="tg-0lax">0.412</td>
    <td class="tg-0lax">0.412</td>
  </tr>
  <tr>
    <td class="tg-0lax">Full MRL</td>
    <td class="tg-0lax"><b>0.514</b></td>
    <td class="tg-0lax"><b>0.491</b></td>
    <td class="tg-0lax"><b>0.464</b></td>
    <td class="tg-0lax"><b>0.435</b></td>
  </tr>
  <tr>
    <td class="tg-0lax"></td>
    <td class="tg-baqh" colspan="4", align="center">Text → Image</td>
  </tr>
  <tr>
    <td class="tg-0lax">CE</td>
    <td class="tg-0lax">0.392</td>
    <td class="tg-0lax">0.364</td>
    <td class="tg-0lax">0.248</td>
    <td class="tg-0lax">0.177</td>
  </tr>
  <tr>
    <td class="tg-0lax">MRL (with &Lscr;<sub>𝓇</sub> only)</td>
    <td class="tg-0lax">0.429</td>
    <td class="tg-0lax">0.389</td>
    <td class="tg-0lax">0.320</td>
    <td class="tg-0lax">0.202</td>
  </tr>
  <tr>
    <td class="tg-0lax">MRL (with &Lscr;<sub>𝒸</sub> only)</td>
    <td class="tg-0lax">0.383</td>
    <td class="tg-0lax">0.382</td>
    <td class="tg-0lax">0.383</td>
    <td class="tg-0lax">0.383</td>
  </tr>
  <tr>
    <td class="tg-0lax">Full MRL</td>
    <td class="tg-0lax"><b>0.461</b></td>
    <td class="tg-0lax"><b>0.453</b></td>
    <td class="tg-0lax"><b>0.421</b></td>
    <td class="tg-0lax"><b>0.400</b></td>
  </tr>
</tbody>
</table>

## Citation
If you find MRL useful in your research, please consider citing:
```
@inproceedings{hu2021MRL,
   title={Learning Cross-Modal Retrieval with Noisy Labels},
   author={Peng Hu, Xi Peng, Hongyuan Zhu, Liangli Zhen, Jie Lin},
   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   pages={5403--5413},
   year={2021}
}
```
