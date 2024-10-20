[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py

Training Results for datasets are as follows.

Simple Dataset:
Epoch: 10/500, loss: 36.013970089486364, correct: 27
Epoch 10 took 0.0367 seconds
Epoch: 20/500, loss: 35.467115909846314, correct: 27
Epoch 20 took 0.0384 seconds
Epoch: 30/500, loss: 35.120580346606424, correct: 27
Epoch 30 took 0.0373 seconds
Epoch: 40/500, loss: 34.89924850423879, correct: 27
Epoch 40 took 0.0360 seconds
Epoch: 50/500, loss: 34.75711290452215, correct: 27
Epoch 50 took 0.0366 seconds
Epoch: 60/500, loss: 34.66549294287236, correct: 27
Epoch 60 took 0.0376 seconds
Epoch: 70/500, loss: 34.606282113171716, correct: 27
Epoch 70 took 0.0386 seconds
Epoch: 80/500, loss: 34.567947481064074, correct: 27
Epoch 80 took 0.0405 seconds
Epoch: 90/500, loss: 34.54309754961732, correct: 27
Epoch 90 took 0.0366 seconds
Epoch: 100/500, loss: 34.52697470311723, correct: 27
Epoch 100 took 0.0384 seconds
Epoch: 110/500, loss: 34.51650751413153, correct: 27
Epoch 110 took 0.0366 seconds
Epoch: 120/500, loss: 34.509709007524776, correct: 27
Epoch 120 took 0.0367 seconds
Epoch: 130/500, loss: 34.5052918939988, correct: 27
Epoch 130 took 0.0363 seconds
Epoch: 140/500, loss: 34.50242132842674, correct: 27
Epoch 140 took 0.0367 seconds
Epoch: 150/500, loss: 34.50055549260701, correct: 27
Epoch 150 took 0.0368 seconds
Epoch: 160/500, loss: 34.49934255872569, correct: 27
Epoch 160 took 0.0404 seconds
Epoch: 170/500, loss: 34.49855398093632, correct: 27
Epoch 170 took 0.0366 seconds
Epoch: 180/500, loss: 34.498041254932716, correct: 27
Epoch 180 took 0.0366 seconds
Epoch: 190/500, loss: 34.4977078654756, correct: 27
Epoch 190 took 0.0373 seconds
Epoch: 200/500, loss: 34.49749107593336, correct: 27
Epoch 200 took 0.0421 seconds
Epoch: 210/500, loss: 34.4973501014964, correct: 27
Epoch 210 took 0.0366 seconds
Epoch: 220/500, loss: 34.49725842571214, correct: 27
Epoch 220 took 0.0374 seconds
Epoch: 230/500, loss: 34.497198807555826, correct: 27
Epoch 230 took 0.0371 seconds
Epoch: 240/500, loss: 34.49716003628094, correct: 27
Epoch 240 took 0.0376 seconds
Epoch: 250/500, loss: 34.49713482193613, correct: 27
Epoch 250 took 0.0370 seconds
Epoch: 260/500, loss: 34.49711842396481, correct: 27
Epoch 260 took 0.0450 seconds
Epoch: 270/500, loss: 34.49710775956506, correct: 27
Epoch 270 took 0.0375 seconds
Epoch: 280/500, loss: 34.49710082393707, correct: 27
Epoch 280 took 0.0368 seconds
Epoch: 290/500, loss: 34.49709631330249, correct: 27
Epoch 290 took 0.0373 seconds
Epoch: 300/500, loss: 34.49709337976612, correct: 27
Epoch 300 took 0.0369 seconds
Epoch: 310/500, loss: 34.49709147190439, correct: 27
Epoch 310 took 0.0371 seconds
Epoch: 320/500, loss: 34.49709023109925, correct: 27
Epoch 320 took 0.0373 seconds
Epoch: 330/500, loss: 34.4970894241219, correct: 27
Epoch 330 took 0.0365 seconds
Epoch: 340/500, loss: 34.497088899290425, correct: 27
Epoch 340 took 0.0370 seconds
Epoch: 350/500, loss: 34.49708855795677, correct: 27
Epoch 350 took 0.0371 seconds
Epoch: 360/500, loss: 34.49708833596397, correct: 27
Epoch 360 took 0.0370 seconds
Epoch: 370/500, loss: 34.49708819158662, correct: 27
Epoch 370 took 0.0370 seconds
Epoch: 380/500, loss: 34.49708809768791, correct: 27
Epoch 380 took 0.0367 seconds
Epoch: 390/500, loss: 34.49708803661897, correct: 27
Epoch 390 took 0.0371 seconds
Epoch: 400/500, loss: 34.497087996901534, correct: 27
Epoch 400 took 0.0371 seconds
Epoch: 410/500, loss: 34.497087971070464, correct: 27
Epoch 410 took 0.0368 seconds
Epoch: 420/500, loss: 34.49708795427063, correct: 27
Epoch 420 took 0.0368 seconds
Epoch: 430/500, loss: 34.49708794334451, correct: 27
Epoch 430 took 0.0368 seconds
Epoch: 440/500, loss: 34.497087936238465, correct: 27
Epoch 440 took 0.0408 seconds
Epoch: 450/500, loss: 34.497087931616946, correct: 27
Epoch 450 took 0.0366 seconds
Epoch: 460/500, loss: 34.497087928611194, correct: 27
Epoch 460 took 0.0371 seconds
Epoch: 470/500, loss: 34.49708792665634, correct: 27
Epoch 470 took 0.0370 seconds
Epoch: 480/500, loss: 34.49708792538499, correct: 27
Epoch 480 took 0.0369 seconds
Epoch: 490/500, loss: 34.49708792455809, correct: 27
Epoch 490 took 0.0387 seconds
Epoch: 500/500, loss: 34.49708792402035, correct: 27
Epoch 500 took 0.0459 seconds
Training completed in 24.19 seconds.

Diag Model:
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 18.0731987642279, correct: 47
Epoch 10 took 0.0387 seconds
Epoch: 20/500, loss: 14.78354213788667, correct: 47
Epoch 20 took 0.0364 seconds
Epoch: 30/500, loss: 12.99243188888586, correct: 47
Epoch 30 took 0.0372 seconds
Epoch: 40/500, loss: 12.032801911400512, correct: 47
Epoch 40 took 0.0364 seconds
Epoch: 50/500, loss: 11.510702167490194, correct: 47
Epoch 50 took 0.0370 seconds
Epoch: 60/500, loss: 11.217043751604026, correct: 47
Epoch 60 took 0.0368 seconds
Epoch: 70/500, loss: 11.04391937930463, correct: 47
Epoch 70 took 0.0365 seconds
Epoch: 80/500, loss: 10.935385170544825, correct: 47
Epoch 80 took 0.0371 seconds
Epoch: 90/500, loss: 10.861974420904758, correct: 47
Epoch 90 took 0.1578 seconds
Epoch: 100/500, loss: 10.80788413954869, correct: 47
Epoch 100 took 0.0377 seconds
Epoch: 110/500, loss: 10.764496113434808, correct: 47
Epoch 110 took 0.0438 seconds
Epoch: 120/500, loss: 10.727041927363665, correct: 47
Epoch 120 took 0.0380 seconds
Epoch: 130/500, loss: 10.692850649291286, correct: 47
Epoch 130 took 0.0373 seconds
Epoch: 140/500, loss: 10.660410077176145, correct: 47
Epoch 140 took 0.0370 seconds
Epoch: 150/500, loss: 10.628855319712473, correct: 47
Epoch 150 took 0.0360 seconds
Epoch: 160/500, loss: 10.597686301778664, correct: 47
Epoch 160 took 0.0364 seconds
Epoch: 170/500, loss: 10.566609988519316, correct: 47
Epoch 170 took 0.0366 seconds
Epoch: 180/500, loss: 10.535451502655063, correct: 47
Epoch 180 took 0.0361 seconds
Epoch: 190/500, loss: 10.504103719331615, correct: 47
Epoch 190 took 0.0365 seconds
Epoch: 200/500, loss: 10.472498535761098, correct: 47
Epoch 200 took 0.0366 seconds
Epoch: 210/500, loss: 10.440590430036575, correct: 47
Epoch 210 took 0.0395 seconds
Epoch: 220/500, loss: 10.40834702129545, correct: 47
Epoch 220 took 0.0362 seconds
Epoch: 230/500, loss: 10.375743632626852, correct: 47
Epoch 230 took 0.0369 seconds
Epoch: 240/500, loss: 10.342760148093758, correct: 47
Epoch 240 took 0.0362 seconds
Epoch: 250/500, loss: 10.309379186926087, correct: 47
Epoch 250 took 0.0366 seconds
Epoch: 260/500, loss: 10.275585034969545, correct: 47
Epoch 260 took 0.0367 seconds
Epoch: 270/500, loss: 10.24136301197961, correct: 47
Epoch 270 took 0.0366 seconds
Epoch: 280/500, loss: 10.206699090072002, correct: 47
Epoch 280 took 0.0360 seconds
Epoch: 290/500, loss: 10.171579657131723, correct: 47
Epoch 290 took 0.0367 seconds
Epoch: 300/500, loss: 10.135991364082022, correct: 47
Epoch 300 took 0.0371 seconds
Epoch: 310/500, loss: 10.099921020837943, correct: 47
Epoch 310 took 0.0365 seconds
Epoch: 320/500, loss: 10.063355520673216, correct: 47
Epoch 320 took 0.0365 seconds
Epoch: 330/500, loss: 10.0262817812983, correct: 47
Epoch 330 took 0.0366 seconds
Epoch: 340/500, loss: 9.988686695876135, correct: 47
Epoch 340 took 0.0368 seconds
Epoch: 350/500, loss: 9.95055709003853, correct: 47
Epoch 350 took 0.0375 seconds
Epoch: 360/500, loss: 9.911879682601164, correct: 47
Epoch 360 took 0.0365 seconds
Epoch: 370/500, loss: 9.87264104862014, correct: 47
Epoch 370 took 0.0364 seconds
Epoch: 380/500, loss: 9.832827583982121, correct: 47
Epoch 380 took 0.0371 seconds
Epoch: 390/500, loss: 9.79242547104147, correct: 47
Epoch 390 took 0.0366 seconds
Epoch: 400/500, loss: 9.7514206450092, correct: 47
Epoch 400 took 0.0365 seconds
Epoch: 410/500, loss: 9.709798760914587, correct: 47
Epoch 410 took 0.0365 seconds
Epoch: 420/500, loss: 9.66754516103355, correct: 47
Epoch 420 took 0.0365 seconds
Epoch: 430/500, loss: 9.624644842726852, correct: 47
Epoch 430 took 0.0366 seconds
Epoch: 440/500, loss: 9.58108242666613, correct: 47
Epoch 440 took 0.0370 seconds
Epoch: 450/500, loss: 9.536842125453497, correct: 47
Epoch 450 took 0.0365 seconds
Epoch: 460/500, loss: 9.491907712664409, correct: 47
Epoch 460 took 0.0369 seconds
Epoch: 470/500, loss: 9.446262492366126, correct: 47
Epoch 470 took 0.0366 seconds
Epoch: 480/500, loss: 9.39988926918712, correct: 47
Epoch 480 took 0.0371 seconds
Epoch: 490/500, loss: 9.352770319036793, correct: 47
Epoch 490 took 0.0365 seconds
Epoch: 500/500, loss: 9.304887360601462, correct: 47
Epoch 500 took 0.0365 seconds
Training completed in 23.26 seconds.

Split Dataset:
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 35.78583034541313, correct: 25
Epoch 10 took 0.0384 seconds
Epoch: 20/500, loss: 35.45023594114807, correct: 25
Epoch 20 took 0.0360 seconds
Epoch: 30/500, loss: 35.22445068339854, correct: 25
Epoch 30 took 0.0362 seconds
Epoch: 40/500, loss: 35.074744010462354, correct: 25
Epoch 40 took 0.0360 seconds
Epoch: 50/500, loss: 34.97277613852801, correct: 25
Epoch 50 took 0.0365 seconds
Epoch: 60/500, loss: 34.902739627341624, correct: 25
Epoch 60 took 0.0379 seconds
Epoch: 70/500, loss: 34.8496646148893, correct: 25
Epoch 70 took 0.0365 seconds
Epoch: 80/500, loss: 34.80460260358876, correct: 24
Epoch 80 took 0.0367 seconds
Epoch: 90/500, loss: 34.76476203340495, correct: 24
Epoch 90 took 0.0366 seconds
Epoch: 100/500, loss: 34.73539328491155, correct: 24
Epoch 100 took 0.0365 seconds
Epoch: 110/500, loss: 34.713677940169916, correct: 25
Epoch 110 took 0.0368 seconds
Epoch: 120/500, loss: 34.69316470132365, correct: 25
Epoch 120 took 0.0402 seconds
Epoch: 130/500, loss: 34.673327365154876, correct: 25
Epoch 130 took 0.0375 seconds
Epoch: 140/500, loss: 34.65372412929977, correct: 25
Epoch 140 took 0.0370 seconds
Epoch: 150/500, loss: 34.63395704903794, correct: 25
Epoch 150 took 0.0366 seconds
Epoch: 160/500, loss: 34.613646821253134, correct: 25
Epoch 160 took 0.0366 seconds
Epoch: 170/500, loss: 34.59260345424165, correct: 19
Epoch 170 took 0.0365 seconds
Epoch: 180/500, loss: 34.57065762734834, correct: 21
Epoch 180 took 0.0365 seconds
Epoch: 190/500, loss: 34.54771719521512, correct: 23
Epoch 190 took 0.0367 seconds
Epoch: 200/500, loss: 34.523324106351524, correct: 25
Epoch 200 took 0.0362 seconds
Epoch: 210/500, loss: 34.49704633004608, correct: 26
Epoch 210 took 0.0363 seconds
Epoch: 220/500, loss: 34.468581309167924, correct: 26
Epoch 220 took 0.0366 seconds
Epoch: 230/500, loss: 34.43872787656965, correct: 27
Epoch 230 took 0.0360 seconds
Epoch: 240/500, loss: 34.40633511401681, correct: 29
Epoch 240 took 0.0364 seconds
Epoch: 250/500, loss: 34.37177241949438, correct: 29
Epoch 250 took 0.0366 seconds
Epoch: 260/500, loss: 34.33548134163268, correct: 31
Epoch 260 took 0.0368 seconds
Epoch: 270/500, loss: 34.29616301922575, correct: 32
Epoch 270 took 0.0366 seconds
Epoch: 280/500, loss: 34.2533897058873, correct: 32
Epoch 280 took 0.0362 seconds
Epoch: 290/500, loss: 34.20806904589521, correct: 32
Epoch 290 took 0.0369 seconds
Epoch: 300/500, loss: 34.16368883592904, correct: 32
Epoch 300 took 0.0366 seconds
Epoch: 310/500, loss: 34.11665588864397, correct: 32
Epoch 310 took 0.0366 seconds
Epoch: 320/500, loss: 34.066310766210705, correct: 33
Epoch 320 took 0.0363 seconds
Epoch: 330/500, loss: 34.012404307433, correct: 34
Epoch 330 took 0.0366 seconds
Epoch: 340/500, loss: 33.95663720561315, correct: 35
Epoch 340 took 0.0368 seconds
Epoch: 350/500, loss: 33.901353475647134, correct: 35
Epoch 350 took 0.0371 seconds
Epoch: 360/500, loss: 33.84375237031983, correct: 35
Epoch 360 took 0.0364 seconds
Epoch: 370/500, loss: 33.78347982902167, correct: 35
Epoch 370 took 0.0365 seconds
Epoch: 380/500, loss: 33.722388908659404, correct: 35
Epoch 380 took 0.0367 seconds
Epoch: 390/500, loss: 33.658158204466815, correct: 35
Epoch 390 took 0.0371 seconds
Epoch: 400/500, loss: 33.5964063308378, correct: 35
Epoch 400 took 0.0365 seconds
Epoch: 410/500, loss: 33.534964950876905, correct: 35
Epoch 410 took 0.0395 seconds
Epoch: 420/500, loss: 33.47149427898367, correct: 35
Epoch 420 took 0.0369 seconds
Epoch: 430/500, loss: 33.405819133674285, correct: 35
Epoch 430 took 0.0365 seconds
Epoch: 440/500, loss: 33.33777108944992, correct: 35
Epoch 440 took 0.0365 seconds
Epoch: 450/500, loss: 33.270227545848, correct: 35
Epoch 450 took 0.0366 seconds
Epoch: 460/500, loss: 33.20134053032303, correct: 35
Epoch 460 took 0.0366 seconds
Epoch: 470/500, loss: 33.13055492093636, correct: 35
Epoch 470 took 0.0365 seconds
Epoch: 480/500, loss: 33.0576262195017, correct: 35
Epoch 480 took 0.0366 seconds
Epoch: 490/500, loss: 32.98357169180642, correct: 35
Epoch 490 took 0.0371 seconds
Epoch: 500/500, loss: 32.90979807038291, correct: 35
Epoch 500 took 0.0373 seconds
Training completed in 23.20 seconds.

XOR Dataset:
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 37.63186131586553, correct: 30
Epoch 10 took 0.0366 seconds
Epoch: 20/500, loss: 36.115852260631954, correct: 30
Epoch 20 took 0.0366 seconds
Epoch: 30/500, loss: 35.21292344658865, correct: 30
Epoch 30 took 0.0376 seconds
Epoch: 40/500, loss: 34.66263701512764, correct: 30
Epoch 40 took 0.0371 seconds
Epoch: 50/500, loss: 34.32015950650476, correct: 30
Epoch 50 took 0.0372 seconds
Epoch: 60/500, loss: 34.1032095850784, correct: 30
Epoch 60 took 0.0367 seconds
Epoch: 70/500, loss: 33.96344422820382, correct: 30
Epoch 70 took 0.0505 seconds
Epoch: 80/500, loss: 33.87211951680043, correct: 30
Epoch 80 took 0.0393 seconds
Epoch: 90/500, loss: 33.81204292692496, correct: 30
Epoch 90 took 0.0366 seconds
Epoch: 100/500, loss: 33.77210753717139, correct: 30
Epoch 100 took 0.0371 seconds
Epoch: 110/500, loss: 33.74525162277026, correct: 30
Epoch 110 took 0.0368 seconds
Epoch: 120/500, loss: 33.726957284191684, correct: 30
Epoch 120 took 0.0367 seconds
Epoch: 130/500, loss: 33.71430596800653, correct: 30
Epoch 130 took 0.0366 seconds
Epoch: 140/500, loss: 33.70539725343005, correct: 30
Epoch 140 took 0.0365 seconds
Epoch: 150/500, loss: 33.69898552771719, correct: 30
Epoch 150 took 0.0368 seconds
Epoch: 160/500, loss: 33.69424987772396, correct: 30
Epoch 160 took 0.0365 seconds
Epoch: 170/500, loss: 33.69064676158186, correct: 30
Epoch 170 took 0.0365 seconds
Epoch: 180/500, loss: 33.68781482796348, correct: 30
Epoch 180 took 0.0373 seconds
Epoch: 190/500, loss: 33.68551296498986, correct: 30
Epoch 190 took 0.0375 seconds
Epoch: 200/500, loss: 33.68357972799862, correct: 30
Epoch 200 took 0.0365 seconds
Epoch: 210/500, loss: 33.681870156041256, correct: 30
Epoch 210 took 0.0366 seconds
Epoch: 220/500, loss: 33.680066967380604, correct: 30
Epoch 220 took 0.0368 seconds
Epoch: 230/500, loss: 33.67838404745032, correct: 30
Epoch 230 took 0.0365 seconds
Epoch: 240/500, loss: 33.67693097901186, correct: 30
Epoch 240 took 0.0365 seconds
Epoch: 250/500, loss: 33.67543166751441, correct: 30
Epoch 250 took 0.0366 seconds
Epoch: 260/500, loss: 33.67395858238666, correct: 30
Epoch 260 took 0.0366 seconds
Epoch: 270/500, loss: 33.67255336877706, correct: 30
Epoch 270 took 0.0366 seconds
Epoch: 280/500, loss: 33.671206156165624, correct: 30
Epoch 280 took 0.0365 seconds
Epoch: 290/500, loss: 33.66990948128488, correct: 30
Epoch 290 took 0.0370 seconds
Epoch: 300/500, loss: 33.66865752806766, correct: 30
Epoch 300 took 0.0368 seconds
Epoch: 310/500, loss: 33.66744528944342, correct: 30
Epoch 310 took 0.0365 seconds
Epoch: 320/500, loss: 33.66622930889092, correct: 30
Epoch 320 took 0.0365 seconds
Epoch: 330/500, loss: 33.66518552104411, correct: 30
Epoch 330 took 0.0371 seconds
Epoch: 340/500, loss: 33.66415659852571, correct: 30
Epoch 340 took 0.0366 seconds
Epoch: 350/500, loss: 33.66350038421089, correct: 30
Epoch 350 took 0.0366 seconds
Epoch: 360/500, loss: 33.662835018935034, correct: 30
Epoch 360 took 0.0364 seconds
Epoch: 370/500, loss: 33.6620402669229, correct: 30
Epoch 370 took 0.0365 seconds
Epoch: 380/500, loss: 33.661262360299034, correct: 30
Epoch 380 took 0.0365 seconds
Epoch: 390/500, loss: 33.66050009303986, correct: 30
Epoch 390 took 0.0366 seconds
Epoch: 400/500, loss: 33.65975230518731, correct: 30
Epoch 400 took 0.0365 seconds
Epoch: 410/500, loss: 33.65901787523992, correct: 30
Epoch 410 took 0.0365 seconds
Epoch: 420/500, loss: 33.65829571436268, correct: 30
Epoch 420 took 0.0365 seconds
Epoch: 430/500, loss: 33.65758476184327, correct: 30
Epoch 430 took 0.0369 seconds
Epoch: 440/500, loss: 33.656883981404974, correct: 30
Epoch 440 took 0.0367 seconds
Epoch: 450/500, loss: 33.656192358112804, correct: 30
Epoch 450 took 0.0370 seconds
Epoch: 460/500, loss: 33.655715369345565, correct: 30
Epoch 460 took 0.0372 seconds
Epoch: 470/500, loss: 33.655255827744334, correct: 30
Epoch 470 took 0.0366 seconds
Epoch: 480/500, loss: 33.6548008923394, correct: 30
Epoch 480 took 0.0368 seconds
Epoch: 490/500, loss: 33.65420283000178, correct: 30
Epoch 490 took 0.0371 seconds
Epoch: 500/500, loss: 33.65359896274881, correct: 30
Epoch 500 took 0.0366 seconds
Training completed in 23.74 seconds.

Circle Dataset:
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 42.015233295176685, correct: 21
Epoch 10 took 0.0370 seconds
Epoch: 20/500, loss: 40.3181632635424, correct: 21
Epoch 20 took 0.0366 seconds
Epoch: 30/500, loss: 38.960611772720576, correct: 21
Epoch 30 took 0.0387 seconds
Epoch: 40/500, loss: 37.88359636349344, correct: 21
Epoch 40 took 0.0405 seconds
Epoch: 50/500, loss: 37.03455521534369, correct: 21
Epoch 50 took 0.0377 seconds
Epoch: 60/500, loss: 36.368391951420854, correct: 21
Epoch 60 took 0.0382 seconds
Epoch: 70/500, loss: 35.847495404271825, correct: 21
Epoch 70 took 0.0483 seconds
Epoch: 80/500, loss: 35.441150458417866, correct: 21
Epoch 80 took 0.0365 seconds
Epoch: 90/500, loss: 35.12465999993258, correct: 21
Epoch 90 took 0.0369 seconds
Epoch: 100/500, loss: 34.87838950219476, correct: 21
Epoch 100 took 0.0368 seconds
Epoch: 110/500, loss: 34.686856057410466, correct: 21
Epoch 110 took 0.0364 seconds
Epoch: 120/500, loss: 34.53792078365305, correct: 29
Epoch 120 took 0.0365 seconds
Epoch: 130/500, loss: 34.42210444789338, correct: 29
Epoch 130 took 0.0366 seconds
Epoch: 140/500, loss: 34.33202443263641, correct: 29
Epoch 140 took 0.0367 seconds
Epoch: 150/500, loss: 34.26194072936325, correct: 29
Epoch 150 took 0.0367 seconds
Epoch: 160/500, loss: 34.207394854776354, correct: 29
Epoch 160 took 0.0366 seconds
Epoch: 170/500, loss: 34.16492538215115, correct: 29
Epoch 170 took 0.0365 seconds
Epoch: 180/500, loss: 34.13184528022754, correct: 29
Epoch 180 took 0.0366 seconds
Epoch: 190/500, loss: 34.1060683880296, correct: 29
Epoch 190 took 0.0365 seconds
Epoch: 200/500, loss: 34.085974565100116, correct: 29
Epoch 200 took 0.0364 seconds
Epoch: 210/500, loss: 34.070305079485124, correct: 29
Epoch 210 took 0.0365 seconds
Epoch: 220/500, loss: 34.05808153112103, correct: 29
Epoch 220 took 0.0366 seconds
Epoch: 230/500, loss: 34.04854304125874, correct: 29
Epoch 230 took 0.0367 seconds
Epoch: 240/500, loss: 34.04109759366225, correct: 29
Epoch 240 took 0.0370 seconds
Epoch: 250/500, loss: 34.03528432979896, correct: 29
Epoch 250 took 0.0365 seconds
Epoch: 260/500, loss: 34.03074431976038, correct: 29
Epoch 260 took 0.0367 seconds
Epoch: 270/500, loss: 34.0271978915936, correct: 29
Epoch 270 took 0.0365 seconds
Epoch: 280/500, loss: 34.024427037022896, correct: 29
Epoch 280 took 0.0372 seconds
Epoch: 290/500, loss: 34.022261748369026, correct: 29
Epoch 290 took 0.0365 seconds
Epoch: 300/500, loss: 34.02056940165384, correct: 29
Epoch 300 took 0.0365 seconds
Epoch: 310/500, loss: 34.01924650169043, correct: 29
Epoch 310 took 0.0365 seconds
Epoch: 320/500, loss: 34.01821225990163, correct: 29
Epoch 320 took 0.0365 seconds
Epoch: 330/500, loss: 34.01740359519167, correct: 29
Epoch 330 took 0.0365 seconds
Epoch: 340/500, loss: 34.01677124051867, correct: 29
Epoch 340 took 0.0367 seconds
Epoch: 350/500, loss: 34.01627670914737, correct: 29
Epoch 350 took 0.0371 seconds
Epoch: 360/500, loss: 34.015889929710724, correct: 29
Epoch 360 took 0.0364 seconds
Epoch: 370/500, loss: 34.01558740188899, correct: 29
Epoch 370 took 0.0366 seconds
Epoch: 380/500, loss: 34.01535075756297, correct: 29
Epoch 380 took 0.0365 seconds
Epoch: 390/500, loss: 34.01516563792207, correct: 29
Epoch 390 took 0.0370 seconds
Epoch: 400/500, loss: 34.01502081687955, correct: 29
Epoch 400 took 0.0365 seconds
Epoch: 410/500, loss: 34.01490751658075, correct: 29
Epoch 410 took 0.0367 seconds
Epoch: 420/500, loss: 34.014818872777184, correct: 29
Epoch 420 took 0.0366 seconds
Epoch: 430/500, loss: 34.01474951716208, correct: 29
Epoch 430 took 0.0365 seconds
Epoch: 440/500, loss: 34.01469525101459, correct: 29
Epoch 440 took 0.0396 seconds
Epoch: 450/500, loss: 34.014652790146386, correct: 29
Epoch 450 took 0.0367 seconds
Epoch: 460/500, loss: 34.01461956554003, correct: 29
Epoch 460 took 0.0368 seconds
Epoch: 470/500, loss: 34.01459356749751, correct: 29
Epoch 470 took 0.0364 seconds
Epoch: 480/500, loss: 34.01457322378694, correct: 29
Epoch 480 took 0.0366 seconds
Epoch: 490/500, loss: 34.014557304361, correct: 29
Epoch 490 took 0.0365 seconds
Epoch: 500/500, loss: 34.01454484684448, correct: 29
Epoch 500 took 0.0365 seconds
Training completed in 23.79 seconds.

Spiral Dataset:
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 35.84109480686414, correct: 25
Epoch 10 took 0.0369 seconds
Epoch: 20/500, loss: 35.559264605597875, correct: 25
Epoch 20 took 0.0365 seconds
Epoch: 30/500, loss: 35.34041284606914, correct: 25
Epoch 30 took 0.0371 seconds
Epoch: 40/500, loss: 35.17523157141892, correct: 25
Epoch 40 took 0.0365 seconds
Epoch: 50/500, loss: 35.04808231530911, correct: 25
Epoch 50 took 0.0365 seconds
Epoch: 60/500, loss: 34.95191588585831, correct: 25
Epoch 60 took 0.0368 seconds
Epoch: 70/500, loss: 34.87644745642217, correct: 25
Epoch 70 took 0.0366 seconds
Epoch: 80/500, loss: 34.81730645352015, correct: 25
Epoch 80 took 0.0366 seconds
Epoch: 90/500, loss: 34.768843356694624, correct: 25
Epoch 90 took 0.0363 seconds
Epoch: 100/500, loss: 34.72915194826275, correct: 25
Epoch 100 took 0.0368 seconds
Epoch: 110/500, loss: 34.6970626876925, correct: 25
Epoch 110 took 0.0365 seconds
Epoch: 120/500, loss: 34.67130264020724, correct: 26
Epoch 120 took 0.0364 seconds
Epoch: 130/500, loss: 34.65302604303211, correct: 26
Epoch 130 took 0.0366 seconds
Epoch: 140/500, loss: 34.63917909637253, correct: 27
Epoch 140 took 0.0365 seconds
Epoch: 150/500, loss: 34.628276599846664, correct: 27
Epoch 150 took 0.0365 seconds
Epoch: 160/500, loss: 34.61952969558545, correct: 27
Epoch 160 took 0.0368 seconds
Epoch: 170/500, loss: 34.61249590220042, correct: 27
Epoch 170 took 0.0364 seconds
Epoch: 180/500, loss: 34.60691614132236, correct: 27
Epoch 180 took 0.0367 seconds
Epoch: 190/500, loss: 34.60225169043493, correct: 27
Epoch 190 took 0.0365 seconds
Epoch: 200/500, loss: 34.59861820515939, correct: 26
Epoch 200 took 0.0367 seconds
Epoch: 210/500, loss: 34.595342908577166, correct: 26
Epoch 210 took 0.0367 seconds
Epoch: 220/500, loss: 34.592678044603325, correct: 25
Epoch 220 took 0.0365 seconds
Epoch: 230/500, loss: 34.59039445817391, correct: 25
Epoch 230 took 0.0368 seconds
Epoch: 240/500, loss: 34.588392728800166, correct: 25
Epoch 240 took 0.0367 seconds
Epoch: 250/500, loss: 34.58659285133448, correct: 25
Epoch 250 took 0.0368 seconds
Epoch: 260/500, loss: 34.58496000195183, correct: 25
Epoch 260 took 0.0368 seconds
Epoch: 270/500, loss: 34.58345419454557, correct: 25
Epoch 270 took 0.0365 seconds
Epoch: 280/500, loss: 34.582034954195606, correct: 25
Epoch 280 took 0.0371 seconds
Epoch: 290/500, loss: 34.580840886454276, correct: 25
Epoch 290 took 0.0366 seconds
Epoch: 300/500, loss: 34.579737540231285, correct: 25
Epoch 300 took 0.0365 seconds
Epoch: 310/500, loss: 34.57872643445627, correct: 25
Epoch 310 took 0.0423 seconds
Epoch: 320/500, loss: 34.57798549255338, correct: 25
Epoch 320 took 0.0365 seconds
Epoch: 330/500, loss: 34.576747623101966, correct: 25
Epoch 330 took 0.0369 seconds
Epoch: 340/500, loss: 34.57580683216256, correct: 25
Epoch 340 took 0.0368 seconds
Epoch: 350/500, loss: 34.575188091004286, correct: 24
Epoch 350 took 0.0365 seconds
Epoch: 360/500, loss: 34.574742653002986, correct: 25
Epoch 360 took 0.0368 seconds
Epoch: 370/500, loss: 34.57429489134602, correct: 25
Epoch 370 took 0.0369 seconds
Epoch: 380/500, loss: 34.57367473068511, correct: 25
Epoch 380 took 0.0365 seconds
Epoch: 390/500, loss: 34.573475415652815, correct: 25
Epoch 390 took 0.0367 seconds
Epoch: 400/500, loss: 34.57347647831251, correct: 25
Epoch 400 took 0.0369 seconds
Epoch: 410/500, loss: 34.57281884668831, correct: 25
Epoch 410 took 0.0366 seconds
Epoch: 420/500, loss: 34.572496236080895, correct: 25
Epoch 420 took 0.0366 seconds
Epoch: 430/500, loss: 34.5721881331808, correct: 25
Epoch 430 took 0.0365 seconds
Epoch: 440/500, loss: 34.5718021269541, correct: 26
Epoch 440 took 0.0365 seconds
Epoch: 450/500, loss: 34.571498893402314, correct: 26
Epoch 450 took 0.0366 seconds
Epoch: 460/500, loss: 34.571209239506814, correct: 26
Epoch 460 took 0.0369 seconds
Epoch: 470/500, loss: 34.570934281883915, correct: 26
Epoch 470 took 0.0369 seconds
Epoch: 480/500, loss: 34.570607658868106, correct: 26
Epoch 480 took 0.0369 seconds
Epoch: 490/500, loss: 34.570491000330094, correct: 26
Epoch 490 took 0.0372 seconds
Epoch: 500/500, loss: 34.56995475597236, correct: 26
Epoch 500 took 0.0373 seconds
Training completed in 23.16 seconds.

