import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# --- 1. DATA INPUT ---
# The raw experiment data provided by the user.
RAW_DATA = """
"Name","Agent","State","Notes","User","Tags","Created","Runtime","Sweep","batch_size","csv_index","data_dir","epochs","eval_batch_size","img_size","lr","method","metric.goal","metric.name","model_type","monitor","monitor_mode","num_patients","num_workers","output_dir","parameters.batch_size","parameters.csv_index","parameters.data_dir","parameters.epochs","parameters.img_size","parameters.lr","parameters.model_type","parameters.model_types","parameters.num_patients","parameters.num_workers","parameters.preprocessed_dir","parameters.wavelets","preprocessed_dir","program","save_dir","save_freq","skip_eval","timing_frequency","wavelet","artifact/name","avg_epoch_loss","backward_time_ms","batch_loss","batch_time_ms","best_epoch","best_loss","best_metric","checkpoint/best_loss","checkpoint/load_path","checkpoint/path","data_fetch_time_ms","data_load_time_ms","data_transfer_time_ms","dataloader/batch_size","dataloader/num_workers","dataloader/persistent_workers","dataloader/pin_memory","dataset/length","dataset/load_time_seconds","dataset/num_samples","epoch","epoch/num_batches","epoch/number","epoch_time_seconds","error/exception","eval/final_timing/avg_data_load_time_ms","eval/final_timing/avg_forward_time_ms","eval/final_timing/avg_metric_time_ms","eval/final_timing/avg_wavelet_time_ms","eval/final_timing/samples_per_second","eval/final_timing/total_evaluation_time_s","eval/final_timing/total_samples","eval/final_timing/total_wavelet_time_s","eval/mse_flair_mean","eval/mse_flair_std","eval/mse_mean","eval/mse_std","eval/mse_t1_mean","eval/mse_t1_std","eval/mse_t1ce_mean","eval/mse_t1ce_std","eval/mse_t2_mean","eval/mse_t2_std","eval/num_samples","eval/progress","eval/running_mse_flair","eval/running_mse_t1","eval/running_mse_t1ce","eval/running_mse_t2","eval/running_ssim_flair","eval/running_ssim_t1","eval/running_ssim_t1ce","eval/running_ssim_t2","eval/ssim_flair_mean","eval/ssim_flair_std","eval/ssim_mean","eval/ssim_std","eval/ssim_t1_mean","eval/ssim_t1_std","eval/ssim_t1ce_mean","eval/ssim_t1ce_std","eval/ssim_t2_mean","eval/ssim_t2_std","eval/timing/avg_forward_time_ms","eval/timing/samples_per_second","final_timing/avg_backward_time_ms","final_timing/avg_batch_time_ms","final_timing/avg_forward_time_ms","final_timing/avg_wavelet_time_ms","final_timing/samples_per_second","final_timing/total_training_minutes","forward_time_ms","inference/avg_data_load_time_ms","inference/avg_forward_time_ms","inference/avg_wavelet_time_ms","inference/samples_per_second","inference_timing/avg_data_load_time_ms","inference_timing/avg_forward_time_ms","inference_timing/avg_wavelet_time_ms","inference_timing/samples_per_second","learning_rate","log/info","system/device","timing/avg_backward_time_ms","timing/avg_batch_time_ms","timing/avg_data_fetch_time_ms","timing/avg_data_load_time_ms","timing/avg_data_transfer_time_ms","timing/avg_dataset_getitem_time_ms","timing/avg_forward_time_ms","timing/avg_wavelet_time_ms","timing/samples_per_second","train_loss","val_loss","wavelet_transform_time_ms"
"unetr_wavelet_db2_1763027298","","failed","-","","","2025-11-13T09:48:11.000Z","9910","1wfhjnkh","8","","","100","16","256","0.0001","","","","unetr","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","db2","unetr_wavelet_db2_1763027298_best_model","0.00032186499478856245","304.0673630312085","0.00078502984251827","4419.481853023171","","0.00032186499478856245","","0.00032186499478856245","./checkpoints/unetr_wavelet_db2_best.pth","./checkpoints/unetr_wavelet_db2_best.pth","0.05536386743187904","1.2625788804143667","1.2072150129824877","8","16","true","true","7440","0.01106119155883789","7440","100","930","100","75.20472717285156","'mse_t1n_mean'","3.669427677247954","80.00213973754917","246.15867884218773","14.867175812833011","199.99465079920063","37.20099497796036","7440","0.1486717581283301","0.0007360102608799934","0.0008398260688409209","0.0007364520570263267","0.0005929218023084104","0.0006158057367429137","0.000609878683462739","0.0007928345003165305","0.000579287763684988","0.000801157730165869","0.0007184187415987253","7440","0.989247311827957","0.0007392634288407862","0.0006193070439621806","0.0007972331950441003","0.000804279581643641","0.936590417467734","0.9352371736645994","0.9119902656016105","0.9236191691133868","0.936835812762842","0.04376125745084095","0.927126105852269","0.04595474009963096","0.935413150404077","0.054501701551898894","0.9123224208439265","0.05738847873665585","0.9239330393982302","0.06694705827914572","80.39082833787995","199.0276792864049","42.55470188284012","175.59738920184392","129.5507006427694","7.070980668067932","45.558763922191574","136.08797663142906","4111.23720696196","5.193704105913639","289.81437840964645","3.969531927723437","27.603875431922734","5.193704105913639","289.81437840964645","3.969531927723437","27.603875431922734","0.0001","============================================================","cuda","42.55098512598204","175.5335597981507","0.5765959925961434","2.3716189134385575","1.795022920842414","0","129.49057213538208","7.070980668067932","45.57533049064435","","","4.57663694396615"
"unetr_wavelet_haar_1763026462","","failed","-","","","2025-11-13T09:34:15.000Z","5902","1wfhjnkh","8","","","100","16","256","0.0001","","","","unetr","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","haar","unetr_wavelet_haar_1763026462_best_model","0.0003613050836476407","226.04526905342937","0.0008812145097181201","711.8072689045221","","0.0003401265937149004","","0.0003401265937149004","./checkpoints/unetr_wavelet_haar_best.pth","./checkpoints/unetr_wavelet_haar_best.pth","0.04739011637866497","1.7452402971684933","1.6978501807898283","8","16","true","true","7440","0.014183282852172852","7440","100","930","100","52.28111505508423","'mse_t1n_mean'","3.2765021432511587","39.12788848492808","222.85207618999303","6.558951805345714","408.915497859363","18.194468145491555","7440","0.06558951805345714","0.0005728754913434386","0.0007965816766954958","0.0006813621730543673","0.0005913004279136658","0.0004941831575706601","0.0005209095543250442","0.000824767688754946","0.0006305240094661713","0.0008336223545484245","0.000762639450840652","7440","0.989247311827957","0.0005764731904491782","0.000497323228046298","0.0008295333827845752","0.0008369339047931135","0.9081846860918136","0.906542749869952","0.8819141452245365","0.9081410506069532","0.9085385595459388","0.06704630404928444","0.9015261492385476","0.05488302318472029","0.9068118081255728","0.06219487955178505","0.8822913208214287","0.06061387803266558","0.9084629084612506","0.06249480650950202","39.16456836978794","408.5325248303416","42.70386306045737","92.76730210815748","46.48006511690416","6.716448841616511","86.23728208321499","71.89465913382203","482.1280960459262","3.80183600820601","45.10983983753249","3.30151847563684","177.34489922404475","3.80183600820601","45.10983983753249","3.30151847563684","177.34489922404475","0.0001","============================================================","cuda","42.70057450948173","92.75750577725093","0.4538340139306272","2.264087667742981","1.8102536538123537","0","46.47351517799418","6.716448841616511","86.24638979848491","","","4.225576063618064"
"unetr_nowavelet_1763025703","","failed","-","","","2025-11-13T09:21:36.000Z","10308","1wfhjnkh","8","","","100","16","256","0.0001","","","","unetr","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","none","unetr_nowavelet_1763025703_best_model","0.003969560807911298","242.5241181626916","0.006034833379089832","76440.23553002626","","0.003938326970123316","","0.003938326970123316","./checkpoints/unetr_baseline_best.pth","./checkpoints/unetr_baseline_best.pth","0.07844716310501099","1.2783741112798452","1.1999269481748345","8","16","true","true","7440","0.01052999496459961","7440","100","930","100","393.55369305610657","'mse_t1n_mean'","3.4246480643188444","619.8808320297269","235.7897904986936","","25.811412731717937","288.244586893823","7440","","0.0005361494258977473","0.0008483165293000638","0.0006769836763851345","0.0006036701379343867","0.0004682050202973187","0.0005702301277779043","0.0008198561263270676","0.0006307549774646759","0.0008837242494337261","0.0007612336194142699","7440","0.989247311827957","0.0005398246576078236","0.0004714098758995533","0.0008247271180152893","0.0008868890581652522","0.9652193159233862","0.967173988269708","0.9486087256126507","0.9624588778430055","0.965388521284113","0.0306889570131092","0.9610276427793096","0.022844825910173176","0.9672974216671636","0.020838108645162487","0.9488001818670444","0.03332597629282577","0.962624446298918","0.033713706775543965","624.9910871879099","25.600365073989348","37.095568260949065","200.32318598113113","160.28407995869475","","39.935467084441925","155.25046913537662","403.95566215738654","5.802666184026748","682.9327167314477","","11.714184727140362","5.802666184026748","682.9327167314477","","11.714184727140362","0.0001","============================================================","cuda","37.08993648283455","198.71574649097732","0.514623675350056","1.8468061819710169","1.3321825066209607","0","158.6821284891351","","38.12448236634148","","",""
"swin_wavelet_db2_1763020588","","failed","-","","","2025-11-13T07:56:21.000Z","5628","1wfhjnkh","8","","","100","16","256","0.0001","","","","swin","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","db2","swin_wavelet_db2_1763020588_best_model","0.0003933805750147189","173.34445822052658","0.0011015022173523905","5959.147579036653","","0.0003722026903477688","","0.0003722026903477688","./checkpoints/swin_wavelet_db2_best.pth","./checkpoints/swin_wavelet_db2_best.pth","0.05181110464036465","1.7805311363190413","1.7287200316786766","8","16","true","true","7440","0.010267972946166992","7440","100","930","100","79.16163086891174","'mse_t1n_mean'","3.6780176829466575","88.6378059228782","219.01106441153155","6.804909883067012","180.50988326494956","41.216579754138365","7440","0.06804909883067012","0.0006191694992594421","0.0008177909767255187","0.0007265754975378513","0.000597848033066839","0.0005252964911051095","0.0005278846947476268","0.0008617629064247012","0.000628648791462183","0.0009000729769468307","0.0007613584166392684","7440","0.989247311827957","0.0006228897836990654","0.000528524222318083","0.000866639253217727","0.0009032816742546856","0.9102412124515716","0.9176946060614936","0.8930724213654034","0.9176631511415247","0.9104467144148464","0.05412614142774051","0.9099408800621612","0.040750139733236414","0.9178833054187442","0.04542437330902709","0.8934543731754746","0.05513268294900754","0.91797912723958","0.05389887552238817","89.06804121739823","179.6379462409758","38.056708632507714","91.73742412733696","50.32321188765608","5.966996031347662","87.20541345149967","71.09650369868613","5782.199547858909","2.5748483021743596","99.96703481068836","3.038788379635662","80.02638084794579","2.5748483021743596","99.96703481068836","3.038788379635662","80.02638084794579","0.0001","============================================================","cuda","38.054569126898365","91.61489559317262","0.37857376745684146","2.091170772414866","1.7125970049580244","0","50.202688703359186","5.966996031347662","87.32204461079122","","","4.485995043069124"
"swin_wavelet_haar_1763019763","","failed","-","","","2025-11-13T07:42:37.000Z","5691","1wfhjnkh","8","","","100","16","256","0.0001","","","","swin","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","haar","swin_wavelet_haar_1763019763_best_model","0.00033508450877527276","170.8005468826741","0.0005952991778030992","912.0667190290987","","0.0003223078460113386","","0.0003223078460113386","./checkpoints/swin_wavelet_haar_best.pth","./checkpoints/swin_wavelet_haar_best.pth","0.07311790250241756","1.985823968425393","1.9127060659229755","8","16","true","true","7440","0.009647607803344728","7440","100","930","100","55.15480589866638","'mse_t1n_mean'","3.9239164151411545","48.47663718035385","237.04025482869037","8.847414795309305","330.0558976579409","22.541636288864535","7440","0.08847414795309305","0.0005618311115540564","0.0007682997384108603","0.0006769027677364647","0.0005410647136159241","0.0004761098825838417","0.0004859619657509029","0.000782401068136096","0.000571110867895186","0.0008872689795680344","0.0006992301787249744","7440","0.989247311827957","0.0005652725230902433","0.000479084555990994","0.0007868437096476555","0.0008898723172023892","0.91128444812181","0.9303434980238496","0.9089308403571136","0.9142452043077988","0.911563415841288","0.0430214918103636","0.9164919738121912","0.04078441801999336","0.9305798597692212","0.04009982805721648","0.9092646923809732","0.0464973211477021","0.9145599272572824","0.05549460187104179","48.54207361114985","329.6109706431018","35.66723548196837","92.40423419396645","51.98281953181891","6.850347637664527","86.57611926318376","71.61328150032399","667.3254100605845","4.219109455589205","56.45460827508941","4.018778887111694","141.7067666295366","4.219109455589205","56.45460827508941","4.018778887111694","141.7067666295366","0.0001","============================================================","cuda","35.66402261052072","92.38816510015717","0.4012009330284669","3.501828198973446","3.100627265944979","0","51.97178376620389","6.850347637664527","86.59117746658647","","","4.25764312967658"
"swin_nowavelet_1763019025","","failed","-","","","2025-11-13T07:30:18.000Z","10195","1wfhjnkh","8","","","100","16","256","0.0001","","","","swin","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","none","swin_nowavelet_1763019025_best_model","0.00397596092924716","255.3537900093943","0.006486466154456139","662.7566970419139","","0.003966099986686341","","0.003966099986686341","./checkpoints/swin_baseline_best.pth","./checkpoints/swin_baseline_best.pth","0.04927092231810093","1.7606390174478292","1.7113680951297283","8","16","true","true","7440","0.009423494338989258","7440","100","930","100","42.45016169548035","'mse_t1n_mean'","3.286282530414962","39.65569408059681","218.88296253959177","","403.472953152235","18.439897747477517","7440","","0.0005619989242404699","0.0008612140663899481","0.0007090289145708084","0.000692891189828515","0.0005230448441579938","0.0006568896933458745","0.0008821007795631886","0.0007865409133955836","0.0008689712267369032","0.0008509930921718478","7440","0.989247311827957","0.0005657724686898291","0.0005265101208351552","0.0008875227067619562","0.0008725189254619181","0.9679000823217688","0.9688647139590792","0.9535220974927798","0.9642283685297172","0.9680537957256492","0.02902540822176102","0.9637735697687196","0.019225024921389108","0.9689685672379528","0.01695698497287548","0.95369211968104","0.028957269781131584","0.9643797964302364","0.03147236521628198","39.68966676229349","403.1275973120675","41.19669341651462","209.81155894184485","165.16072656629328","","38.129453116629406","162.60395817992975","403.95566215738654","3.432254029903561","43.320982221048325","","184.6680197410909","3.432254029903561","43.320982221048325","","184.6680197410909","0.0001","============================================================","cuda","41.192788434215814","209.8389146146904","0.33122264571324905","2.13209308634347","1.800870440630221","0","165.19188327754048","","38.12448236634148","","",""
"unet_wavelet_db2_1763016794","","failed","-","","","2025-11-13T06:53:08.000Z","2722","1wfhjnkh","8","","","100","16","256","0.0001","","","","unet","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","db2","unet_wavelet_db2_1763016794_best_model","0.0004695584360379926","85.16308199614286","0.0007663429714739323","191.75558211281896","","0.0004695584360379926","","0.0004695584360379926","./checkpoints/unet_wavelet_db2_best.pth","./checkpoints/unet_wavelet_db2_best.pth","0.05574501119554043","1.7550401389598846","1.6992951277643442","8","16","true","true","7440","0.010000944137573242","7440","100","930","100","24.464545726776123","'mse_t1n_mean'","2.909235810480451","14.986895476918548","232.5109673754102","6.7639288026839495","1067.5993586958516","6.9689063967671245","7440","0.0676392880268395","0.0007206319714896381","0.0008744815131649375","0.000803464325144887","0.000698284653481096","0.0006691735470667481","0.0007362684700638056","0.0009042779565788804","0.0007458606851287186","0.0009197739418596028","0.0008242462063208222","7440","0.989247311827957","0.0007249896298162639","0.0006734406342729926","0.0009093757253140212","0.0009229169227182864","0.939815619773106","0.949900356909146","0.9357538253208052","0.9521432682517388","0.9399947691130702","0.028728809802151586","0.9445566928604427","0.022472216055893345","0.9500090407372728","0.020596572955137706","0.9359467270177751","0.03197568877935261","0.9522762345736532","0.03075278160573479","15.005146112361691","1066.3008464022032","10.223764349921778","29.69400623626907","12.119857072184285","6.084991681855172","269.4145047584749","23.0128663843032","104.06276304274796","3.6840187502093618","19.556146427057683","3.8325111428275704","409.07854877437825","3.6840187502093618","19.556146427057683","3.8325111428275704","409.07854877437825","0.0001","============================================================","cuda","10.222552973040068","29.691346026973264","0.4911771244149268","2.136587754209451","1.6454106297945243","0","17.007203949964158","6.084991681855172","269.4387783138008","","","3.8791028782725334"
"unet_wavelet_haar_1763016085","","failed","-","","","2025-11-13T06:41:19.000Z","2693","1wfhjnkh","8","","","100","16","256","0.0001","","","","unet","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","haar","unet_wavelet_haar_1763016085_best_model","0.0004877653727758556","84.00258212350309","0.0007525443797931075","196.35763787664473","","0.0004678140913877356","","0.0004678140913877356","./checkpoints/unet_wavelet_haar_best.pth","./checkpoints/unet_wavelet_haar_best.pth","0.047008972615003586","1.7078767996281383","1.660867827013135","8","16","true","true","7440","0.010149240493774414","7440","100","930","100","23.197274208068848","'mse_t1n_mean'","3.5507456269315494","13.262866477492036","220.84664325129685","7.363256928510964","1206.3757127581027","6.167232912033796","7440","0.07363256928510964","0.0008739384356886148","0.0012137395096942782","0.000923743878956884","0.0008785945829004049","0.0008107745670713484","0.000918336445465684","0.0010157863143831491","0.000865479523781687","0.0009944761404767632","0.0008861463866196573","7440","0.989247311827957","0.0008797681075520813","0.000816224841400981","0.0010218985844403503","0.0009982534684240818","0.9476870813851352","0.9369968764058816","0.9335751043626124","0.91458115661564","0.947873202218913","0.030021567426982785","0.933379125372127","0.02151350793373112","0.9371277813932822","0.01811801502592381","0.9337793479762104","0.031467737620035276","0.9147361699001022","0.031313486848165616","13.27281113190742","1205.4718356939852","10.731648844095968","29.30240139641088","15.760092810552406","6.033818339928985","273.0151666334038","22.709361082218432","109.7612800076604","4.006020554807037","17.65047865221277","3.216552590020001","453.2454987557561","4.006020554807037","17.65047865221277","3.216552590020001","453.2454987557561","0.0001","============================================================","cuda","10.730541745536812","29.300266132325667","0.5870759820375494","2.4181559272047153","1.831079945167166","0","15.758939411884272","6.033818339928985","273.0350626806751","","","4.793472122400999"
"unet_nowavelet_1763014681","","failed","-","","","2025-11-13T06:17:54.000Z","1865","1wfhjnkh","8","","","100","16","256","0.0001","","","","unet","","","","16","./checkpoints","","","","","","","","","","","","","./preprocessed_slices","","","","false","50","none","unet_nowavelet_1763014681_best_model","0.00528949900500236","89.13956605829298","0.009666180238127708","199.42262000404295","","0.005087401258248476","","0.005087401258248476","./checkpoints/unet_baseline_best.pth","./checkpoints/unet_baseline_best.pth","0.0538211315870285","2.9536341316998005","2.899813000112772","8","16","true","true","7440","0.01041579246520996","7440","100","930","100","17.18492555618286","'mse_t1n_mean'","2.865756705142958","12.365588937856018","237.32911504514675","","1293.913300887562","5.749998856103048","7440","","0.0010257787071168425","0.0013026598608121276","0.0011379767674952743","0.0010900234337896109","0.0011376756010577085","0.0012292073806747794","0.0014817516785115004","0.0013185779098421335","0.0009067008504644036","0.000863193825352937","7440","0.989247311827957","0.0010323143796995282","0.0011452381731942296","0.001491545932367444","0.000910012109670788","0.9566270404015113","0.9687500697292448","0.953412878885642","0.9661722429357024","0.956772942660436","0.026500730720754512","0.9613687278623214","0.017884285893240825","0.9688353151033096","0.016146068545654344","0.9535629184178636","0.030088373484551484","0.9663037352676768","0.02947591167190254","12.379098654807486","1292.5012108039318","16.020331705685066","30.99400623626907","12.119857072184285","","258.1144218341942","24.020354833108527","106.71991319395602","5.606407772284001","10.486341591458768","","762.8971391239136","5.606407772284001","10.486341591458768","","762.8971391239136","0.0001","============================================================","cuda","16.018987448084218","30.991276986680017","0.5472944588311671","2.576399543377317","2.02910508454615","0","12.118496801073883","","258.13715270391674","","",""
"""

# Define the columns we are interested in for analysis and plotting
METRIC_COLUMNS = [
    'eval/mse_mean', 'eval/ssim_mean', 
    'eval/mse_flair_mean', 'eval/mse_t1_mean', 'eval/mse_t1ce_mean', 'eval/mse_t2_mean',
    'eval/ssim_flair_mean', 'eval/ssim_t1_mean', 'eval/ssim_t1ce_mean', 'eval/ssim_t2_mean',
    'wavelet', 'model_type', 'Runtime'
]

# Define constants for file organization
MODALITIES = ['flair', 't1', 't1ce', 't2']
MODALITY_MSE_COLS = [f'eval/mse_{m}_mean' for m in MODALITIES]
MODALITY_SSIM_COLS = [f'eval/ssim_{m}_mean' for m in MODALITIES]
FIGURE_DIR = 'figures'
REPORT_FILE = 'analysis_report.md'

# --- 2. DATA PROCESSING ---
try:
    df = pd.read_csv(io.StringIO(RAW_DATA))
    df = df[METRIC_COLUMNS].copy()

    # Convert all metric columns to numeric
    for col in df.columns:
        if 'eval/' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where critical metrics are missing
    df.dropna(subset=['eval/mse_mean', 'eval/ssim_mean'], inplace=True)

    # Clean up model_type and wavelet names
    df['model_type'] = df['model_type'].str.upper()
    df['wavelet'] = df['wavelet'].replace({'none': 'Baseline (None)'})
    df['Configuration'] = df['model_type'] + ' + ' + df['wavelet']

except Exception as e:
    print(f"Error processing data: {e}")
    # Use a dummy DataFrame if processing fails to allow the script to exit gracefully
    df = pd.DataFrame(columns=METRIC_COLUMNS + ['Configuration'])

# --- 3. ANALYSIS FUNCTIONS ---

def get_best_performer(df, metric, sort_order):
    """Finds the top performing configuration based on a given metric."""
    if df.empty:
        return {'model': 'N/A', 'wavelet': 'N/A', 'metric_value': np.nan, 'runtime': np.nan}
        
    if sort_order == 'min':
        best_row = df.loc[df[metric].idxmin()]
    else: # max
        best_row = df.loc[df[metric].idxmax()]
        
    return {
        'model': best_row['model_type'],
        'wavelet': best_row['wavelet'],
        'metric_value': best_row[metric],
        'runtime': best_row['Runtime']
    }

def generate_analysis_report(df):
    """Generates the text-based performance report in Markdown format."""
    if df.empty:
        return "# Analysis Failed\nDataframe is empty or could not be processed."
        
    report = ["# Deep Learning Model Performance Analysis\n"]

    # 3.1. Overall Best Performers
    best_mse = get_best_performer(df, 'eval/mse_mean', 'min')
    best_ssim = get_best_performer(df, 'eval/ssim_mean', 'max')
    
    report.append("## OVERALL BEST PERFORMERS (Aggregated Mean)")
    report.append(f"| Metric | Model + Wavelet | Value | Runtime (s) |")
    report.append(f"|:---|:---|:---|:---|")
    report.append(f"| Lowest MSE | {best_mse['model']} + {best_mse['wavelet']} | {best_mse['metric_value']:.6f} | {best_mse['runtime']:.0f} |")
    report.append(f"| Highest SSIM | {best_ssim['model']} + {best_ssim['wavelet']} | {best_ssim['metric_value']:.4f} | {best_ssim['runtime']:.0f} |")
    report.append("\n")

    # 3.2. Baseline Performance Comparison
    baseline_df = df[df['wavelet'] == 'Baseline (None)']
    report.append("## BASELINE PERFORMANCE (No Wavelet - 'Baseline (None)')")
    report.append(baseline_df[['model_type', 'eval/mse_mean', 'eval/ssim_mean', 'Runtime']]
                  .sort_values(by='eval/mse_mean')
                  .to_markdown(index=False, floatfmt=".6f"))
    report.append("\n")

    # 3.3. Wavelet Type Effectiveness
    wavelet_comparison = []
    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model].sort_values(by='eval/mse_mean', ascending=True)
        if model_df.empty: continue
            
        best_w = model_df.iloc[0]
        baseline_w = model_df[model_df['wavelet'] == 'Baseline (None)']
        baseline_mse = baseline_w['eval/mse_mean'].iloc[0] if not baseline_w.empty else np.nan
        
        improvement = np.nan
        if not np.isnan(baseline_mse) and baseline_mse != 0:
            improvement = ((baseline_mse - best_w['eval/mse_mean']) / baseline_mse) * 100
        
        wavelet_comparison.append({
            'Model': model,
            'Best Wavelet': best_w['wavelet'],
            'Best MSE': best_w['eval/mse_mean'],
            'Baseline MSE': baseline_mse,
            'Improvement (vs Baseline)': improvement
        })

    report.append("## BEST WAVELET CONFIGURATION PER MODEL")
    wavelet_comp_df = pd.DataFrame(wavelet_comparison).sort_values(by='Best MSE', ascending=True)
    wavelet_comp_df['Improvement (vs Baseline)'] = wavelet_comp_df['Improvement (vs Baseline)'].apply(
        lambda x: f"{x:.2f}%" if not np.isnan(x) else "N/A"
    )
    report.append(wavelet_comp_df.to_markdown(index=False, floatfmt=(".0f", ".6f", ".6f", ".6f", ".s")))
    report.append("\n")

    # 3.4. Modality-Specific Best Performers
    report.append("## MODALITY-SPECIFIC BEST PERFORMERS (Lowest MSE)")
    report.append(f"| Modality | Best Configuration | MSE | SSIM |")
    report.append(f"|:---|:---|:---|:---|")
    
    for modality in MODALITIES:
        mse_col = f'eval/mse_{modality}_mean'
        ssim_col = f'eval/ssim_{modality}_mean'
        best_row = df.loc[df[mse_col].idxmin()]
        
        report.append(f"| {modality.upper()} | {best_row['model_type']} + {best_row['wavelet']} | {best_row[mse_col]:.6f} | {best_row[ssim_col]:.4f} |")

    report.append("\n")
    
    return "\n".join(report)

# --- 4. PLOTTING FUNCTIONS ---

def plot_combined_metric_comparison(df, metric_col, title, filename, ascending):
    """Figure 1 & 2: Bar chart comparing all Model + Wavelet combinations."""
    plt.figure(figsize=(12, 7))
    df_sorted = df.sort_values(by=metric_col, ascending=ascending)
    colors = plt.cm.get_cmap('viridis', len(df_sorted))
    plt.bar(df_sorted['Configuration'], df_sorted[metric_col], color=colors(np.arange(len(df_sorted))))
    
    metric_label = metric_col.split('/')[1].replace('_mean', '').upper()
    
    plt.title(title)
    plt.ylabel(metric_label)
    plt.xlabel('Configuration (Model + Wavelet)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\n[Figure generated and saved locally to: {filename}]")

def plot_modality_heatmap(df, metric_cols, metric_type, filename, title):
    """Figure 3 & 4: Heatmap of average metric (MSE/SSIM) by Wavelet and Modality (Averaged over all Models)."""
    
    df_pivot = df.groupby('wavelet')[metric_cols].mean()
    df_pivot.columns = [col.split('_')[1].upper() for col in df_pivot.columns]
    
    plt.figure(figsize=(10, 6))
    
    if metric_type == 'MSE':
        # Lower is better -> sort ascending, use reversed colormap
        cmap_to_use = 'RdYlGn_r' 
        df_pivot['avg'] = df_pivot.mean(axis=1)
        df_pivot = df_pivot.sort_values(by='avg', ascending=True).drop(columns='avg')
        fmt = ".6f"
    else: # SSIM
        # Higher is better -> sort descending, use normal colormap
        cmap_to_use = 'RdYlGn'
        df_pivot['avg'] = df_pivot.mean(axis=1)
        df_pivot = df_pivot.sort_values(by='avg', ascending=False).drop(columns='avg')
        fmt = ".4f"

    sns.heatmap(df_pivot, annot=True, fmt=fmt, 
                cmap=cmap_to_use, cbar_kws={'label': f'Mean {metric_type} Score'}, 
                linewidths=.5, linecolor='black')
    
    plt.title(title, pad=15)
    plt.xlabel('Modality')
    plt.ylabel('Wavelet Configuration')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\n[Figure generated and saved locally to: {filename}]")

def plot_model_wavelet_grouped(df, metric_col, title, filename, ylabel, ascending):
    """Figure 5 & 6: Grouped bar chart comparing wavelets within each model."""
    pivot = df.pivot_table(index='wavelet', columns='model_type', values=metric_col)
    
    # Sort the index (wavelets) by the mean performance of the wavelet across all models
    pivot['mean_metric'] = pivot.mean(axis=1)
    pivot_sorted = pivot.sort_values(by='mean_metric', ascending=ascending).drop(columns='mean_metric')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot_sorted.plot(kind='bar', ax=ax)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Wavelet Type')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\n[Figure generated and saved locally to: {filename}]")


# --- 5. EXECUTION ---
if __name__ == '__main__':
    
    if df.empty:
        print("Script terminated due to empty/failed data processing.")
    else:
        # Create Output Directories
        os.makedirs(FIGURE_DIR, exist_ok=True)
        
        # 5.1. Generate Textual Report and Save
        analysis_report = generate_analysis_report(df)
        with open(REPORT_FILE, 'w') as f:
            f.write(analysis_report)
        print(f"\n[Analysis Report saved to: {REPORT_FILE}]")
        print("\n" + "="*80)
        print("ANALYSIS REPORT CONTENT")
        print("="*80)
        print(analysis_report)
        
        # 5.2. Generate 6 Plot Figures
        print("\n" + "="*80)
        print(f"Generating 6 Plot Figures in the '{FIGURE_DIR}' directory")
        print("="*80)
        
        # Figure 1: Overall MSE Comparison (Bar Chart)
        plot_combined_metric_comparison(
            df, 
            'eval/mse_mean', 
            'Figure 1: Overall Mean Squared Error (MSE) Comparison', 
            f'{FIGURE_DIR}/fig1_overall_mse.png', 
            ascending=True
        )
        
        # Figure 2: Overall SSIM Comparison (Bar Chart)
        plot_combined_metric_comparison(
            df, 
            'eval/ssim_mean', 
            'Figure 2: Overall Structural Similarity Index (SSIM) Comparison', 
            f'{FIGURE_DIR}/fig2_overall_ssim.png', 
            ascending=False
        )
        
        # Figure 3: MSE Modality Heatmap
        plot_modality_heatmap(
            df,
            MODALITY_MSE_COLS,
            'MSE',
            f'{FIGURE_DIR}/fig3_modality_mse_heatmap.png',
            'Figure 3: Modality MSE Heatmap (Wavelet vs. Modality, Average over Models)'
        )

        # Figure 4: SSIM Modality Heatmap
        plot_modality_heatmap(
            df,
            MODALITY_SSIM_COLS,
            'SSIM',
            f'{FIGURE_DIR}/fig4_modality_ssim_heatmap.png',
            'Figure 4: Modality SSIM Heatmap (Wavelet vs. Modality, Average over Models)'
        )
        
        # Figure 5: Grouped MSE Comparison by Model
        plot_model_wavelet_grouped(
            df, 
            'eval/mse_mean', 
            'Figure 5: MSE Comparison by Model and Wavelet Type', 
            f'{FIGURE_DIR}/fig5_grouped_mse.png', 
            'Mean Squared Error (MSE)', 
            ascending=True
        )

        # Figure 6: Grouped SSIM Comparison by Model
        plot_model_wavelet_grouped(
            df, 
            'eval/ssim_mean', 
            'Figure 6: SSIM Comparison by Model and Wavelet Type', 
            f'{FIGURE_DIR}/fig6_grouped_ssim.png', 
            'Structural Similarity Index (SSIM)', 
            ascending=False
        )
        
        print("\nAnalysis complete. Check the report and figures in the generated file paths.")