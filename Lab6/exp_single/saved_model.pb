³	
Ñ¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.02v2.9.0-rc2-42-g8a20d54a3c18¢·
¤
&Adam/module_wrapper_3/wyjsciowa/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/module_wrapper_3/wyjsciowa/bias/v

:Adam/module_wrapper_3/wyjsciowa/bias/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_3/wyjsciowa/bias/v*
_output_shapes
:*
dtype0
¬
(Adam/module_wrapper_3/wyjsciowa/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*9
shared_name*(Adam/module_wrapper_3/wyjsciowa/kernel/v
¥
<Adam/module_wrapper_3/wyjsciowa/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/module_wrapper_3/wyjsciowa/kernel/v*
_output_shapes

:
*
dtype0
 
$Adam/module_wrapper_2/ukryta2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/module_wrapper_2/ukryta2/bias/v

8Adam/module_wrapper_2/ukryta2/bias/v/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_2/ukryta2/bias/v*
_output_shapes
:
*
dtype0
¨
&Adam/module_wrapper_2/ukryta2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&Adam/module_wrapper_2/ukryta2/kernel/v
¡
:Adam/module_wrapper_2/ukryta2/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_2/ukryta2/kernel/v*
_output_shapes

:
*
dtype0
 
$Adam/module_wrapper_1/ukryta1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_1/ukryta1/bias/v

8Adam/module_wrapper_1/ukryta1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/ukryta1/bias/v*
_output_shapes
:*
dtype0
¨
&Adam/module_wrapper_1/ukryta1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&Adam/module_wrapper_1/ukryta1/kernel/v
¡
:Adam/module_wrapper_1/ukryta1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_1/ukryta1/kernel/v*
_output_shapes

:
*
dtype0

#Adam/module_wrapper/ukryta_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/module_wrapper/ukryta_1/bias/v

7Adam/module_wrapper/ukryta_1/bias/v/Read/ReadVariableOpReadVariableOp#Adam/module_wrapper/ukryta_1/bias/v*
_output_shapes
:
*
dtype0
¦
%Adam/module_wrapper/ukryta_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*6
shared_name'%Adam/module_wrapper/ukryta_1/kernel/v

9Adam/module_wrapper/ukryta_1/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper/ukryta_1/kernel/v*
_output_shapes

:	
*
dtype0
¤
&Adam/module_wrapper_3/wyjsciowa/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&Adam/module_wrapper_3/wyjsciowa/bias/m

:Adam/module_wrapper_3/wyjsciowa/bias/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_3/wyjsciowa/bias/m*
_output_shapes
:*
dtype0
¬
(Adam/module_wrapper_3/wyjsciowa/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*9
shared_name*(Adam/module_wrapper_3/wyjsciowa/kernel/m
¥
<Adam/module_wrapper_3/wyjsciowa/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/module_wrapper_3/wyjsciowa/kernel/m*
_output_shapes

:
*
dtype0
 
$Adam/module_wrapper_2/ukryta2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*5
shared_name&$Adam/module_wrapper_2/ukryta2/bias/m

8Adam/module_wrapper_2/ukryta2/bias/m/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_2/ukryta2/bias/m*
_output_shapes
:
*
dtype0
¨
&Adam/module_wrapper_2/ukryta2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&Adam/module_wrapper_2/ukryta2/kernel/m
¡
:Adam/module_wrapper_2/ukryta2/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_2/ukryta2/kernel/m*
_output_shapes

:
*
dtype0
 
$Adam/module_wrapper_1/ukryta1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/module_wrapper_1/ukryta1/bias/m

8Adam/module_wrapper_1/ukryta1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/module_wrapper_1/ukryta1/bias/m*
_output_shapes
:*
dtype0
¨
&Adam/module_wrapper_1/ukryta1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*7
shared_name(&Adam/module_wrapper_1/ukryta1/kernel/m
¡
:Adam/module_wrapper_1/ukryta1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/module_wrapper_1/ukryta1/kernel/m*
_output_shapes

:
*
dtype0

#Adam/module_wrapper/ukryta_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Adam/module_wrapper/ukryta_1/bias/m

7Adam/module_wrapper/ukryta_1/bias/m/Read/ReadVariableOpReadVariableOp#Adam/module_wrapper/ukryta_1/bias/m*
_output_shapes
:
*
dtype0
¦
%Adam/module_wrapper/ukryta_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*6
shared_name'%Adam/module_wrapper/ukryta_1/kernel/m

9Adam/module_wrapper/ukryta_1/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/module_wrapper/ukryta_1/kernel/m*
_output_shapes

:	
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

module_wrapper_3/wyjsciowa/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!module_wrapper_3/wyjsciowa/bias

3module_wrapper_3/wyjsciowa/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_3/wyjsciowa/bias*
_output_shapes
:*
dtype0

!module_wrapper_3/wyjsciowa/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*2
shared_name#!module_wrapper_3/wyjsciowa/kernel

5module_wrapper_3/wyjsciowa/kernel/Read/ReadVariableOpReadVariableOp!module_wrapper_3/wyjsciowa/kernel*
_output_shapes

:
*
dtype0

module_wrapper_2/ukryta2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*.
shared_namemodule_wrapper_2/ukryta2/bias

1module_wrapper_2/ukryta2/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_2/ukryta2/bias*
_output_shapes
:
*
dtype0

module_wrapper_2/ukryta2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*0
shared_name!module_wrapper_2/ukryta2/kernel

3module_wrapper_2/ukryta2/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_2/ukryta2/kernel*
_output_shapes

:
*
dtype0

module_wrapper_1/ukryta1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namemodule_wrapper_1/ukryta1/bias

1module_wrapper_1/ukryta1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/ukryta1/bias*
_output_shapes
:*
dtype0

module_wrapper_1/ukryta1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*0
shared_name!module_wrapper_1/ukryta1/kernel

3module_wrapper_1/ukryta1/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper_1/ukryta1/kernel*
_output_shapes

:
*
dtype0

module_wrapper/ukryta_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_namemodule_wrapper/ukryta_1/bias

0module_wrapper/ukryta_1/bias/Read/ReadVariableOpReadVariableOpmodule_wrapper/ukryta_1/bias*
_output_shapes
:
*
dtype0

module_wrapper/ukryta_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	
*/
shared_name module_wrapper/ukryta_1/kernel

2module_wrapper/ukryta_1/kernel/Read/ReadVariableOpReadVariableOpmodule_wrapper/ukryta_1/kernel*
_output_shapes

:	
*
dtype0

NoOpNoOp
êM
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥M
valueMBM BM
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module*

	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_module*

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_module*
<
*0
+1
,2
-3
.4
/5
06
17*
<
*0
+1
,2
-3
.4
/5
06
17*
* 
°
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
7trace_0
8trace_1
9trace_2
:trace_3* 
6
;trace_0
<trace_1
=trace_2
>trace_3* 
* 
ä
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate*m¦+m§,m¨-m©.mª/m«0m¬1m­*v®+v¯,v°-v±.v²/v³0v´1vµ*

Dserving_default* 

*0
+1*

*0
+1*
* 

Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Jtrace_0
Ktrace_1* 

Ltrace_0
Mtrace_1* 
¦
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

*kernel
+bias*

,0
-1*

,0
-1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ytrace_0
Ztrace_1* 

[trace_0
\trace_1* 
¦
]trainable_variables
^	variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

,kernel
-bias*

.0
/1*

.0
/1*
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

htrace_0
itrace_1* 

jtrace_0
ktrace_1* 
¦
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

.kernel
/bias*

00
11*

00
11*
* 

rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*

wtrace_0
xtrace_1* 

ytrace_0
ztrace_1* 
§
{trainable_variables
|	variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

0kernel
1bias*
^X
VARIABLE_VALUEmodule_wrapper/ukryta_1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEmodule_wrapper/ukryta_1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmodule_wrapper_1/ukryta1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmodule_wrapper_1/ukryta1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmodule_wrapper_2/ukryta2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEmodule_wrapper_2/ukryta2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!module_wrapper_3/wyjsciowa/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEmodule_wrapper_3/wyjsciowa/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*

*0
+1*
* 

layers
metrics
Ntrainable_variables
O	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

,0
-1*

,0
-1*
* 

layers
metrics
]trainable_variables
^	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

.0
/1*

.0
/1*
* 

layers
metrics
ltrainable_variables
m	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

00
11*

00
11*
* 

layers
metrics
{trainable_variables
|	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count
 
_fn_kwargs*
M
¡	variables
¢	keras_api

£total

¤count
¥
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

£0
¤1*

¡	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
{
VARIABLE_VALUE%Adam/module_wrapper/ukryta_1/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/module_wrapper/ukryta_1/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_1/ukryta1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_1/ukryta1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_2/ukryta2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_2/ukryta2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/module_wrapper_3/wyjsciowa/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_3/wyjsciowa/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/module_wrapper/ukryta_1/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE#Adam/module_wrapper/ukryta_1/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_1/ukryta1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_1/ukryta1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_2/ukryta2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE$Adam/module_wrapper_2/ukryta2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUE(Adam/module_wrapper_3/wyjsciowa/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE&Adam/module_wrapper_3/wyjsciowa/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

$serving_default_module_wrapper_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ	
Ñ
StatefulPartitionedCallStatefulPartitionedCall$serving_default_module_wrapper_inputmodule_wrapper/ukryta_1/kernelmodule_wrapper/ukryta_1/biasmodule_wrapper_1/ukryta1/kernelmodule_wrapper_1/ukryta1/biasmodule_wrapper_2/ukryta2/kernelmodule_wrapper_2/ukryta2/bias!module_wrapper_3/wyjsciowa/kernelmodule_wrapper_3/wyjsciowa/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_197654
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ç
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename2module_wrapper/ukryta_1/kernel/Read/ReadVariableOp0module_wrapper/ukryta_1/bias/Read/ReadVariableOp3module_wrapper_1/ukryta1/kernel/Read/ReadVariableOp1module_wrapper_1/ukryta1/bias/Read/ReadVariableOp3module_wrapper_2/ukryta2/kernel/Read/ReadVariableOp1module_wrapper_2/ukryta2/bias/Read/ReadVariableOp5module_wrapper_3/wyjsciowa/kernel/Read/ReadVariableOp3module_wrapper_3/wyjsciowa/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9Adam/module_wrapper/ukryta_1/kernel/m/Read/ReadVariableOp7Adam/module_wrapper/ukryta_1/bias/m/Read/ReadVariableOp:Adam/module_wrapper_1/ukryta1/kernel/m/Read/ReadVariableOp8Adam/module_wrapper_1/ukryta1/bias/m/Read/ReadVariableOp:Adam/module_wrapper_2/ukryta2/kernel/m/Read/ReadVariableOp8Adam/module_wrapper_2/ukryta2/bias/m/Read/ReadVariableOp<Adam/module_wrapper_3/wyjsciowa/kernel/m/Read/ReadVariableOp:Adam/module_wrapper_3/wyjsciowa/bias/m/Read/ReadVariableOp9Adam/module_wrapper/ukryta_1/kernel/v/Read/ReadVariableOp7Adam/module_wrapper/ukryta_1/bias/v/Read/ReadVariableOp:Adam/module_wrapper_1/ukryta1/kernel/v/Read/ReadVariableOp8Adam/module_wrapper_1/ukryta1/bias/v/Read/ReadVariableOp:Adam/module_wrapper_2/ukryta2/kernel/v/Read/ReadVariableOp8Adam/module_wrapper_2/ukryta2/bias/v/Read/ReadVariableOp<Adam/module_wrapper_3/wyjsciowa/kernel/v/Read/ReadVariableOp:Adam/module_wrapper_3/wyjsciowa/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_198044
¦

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemodule_wrapper/ukryta_1/kernelmodule_wrapper/ukryta_1/biasmodule_wrapper_1/ukryta1/kernelmodule_wrapper_1/ukryta1/biasmodule_wrapper_2/ukryta2/kernelmodule_wrapper_2/ukryta2/bias!module_wrapper_3/wyjsciowa/kernelmodule_wrapper_3/wyjsciowa/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcount%Adam/module_wrapper/ukryta_1/kernel/m#Adam/module_wrapper/ukryta_1/bias/m&Adam/module_wrapper_1/ukryta1/kernel/m$Adam/module_wrapper_1/ukryta1/bias/m&Adam/module_wrapper_2/ukryta2/kernel/m$Adam/module_wrapper_2/ukryta2/bias/m(Adam/module_wrapper_3/wyjsciowa/kernel/m&Adam/module_wrapper_3/wyjsciowa/bias/m%Adam/module_wrapper/ukryta_1/kernel/v#Adam/module_wrapper/ukryta_1/bias/v&Adam/module_wrapper_1/ukryta1/kernel/v$Adam/module_wrapper_1/ukryta1/bias/v&Adam/module_wrapper_2/ukryta2/kernel/v$Adam/module_wrapper_2/ukryta2/bias/v(Adam/module_wrapper_3/wyjsciowa/kernel/v&Adam/module_wrapper_3/wyjsciowa/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_198159ßû
ß
á
F__inference_sequential_layer_call_and_return_conditional_losses_197601
module_wrapper_input'
module_wrapper_197580:	
#
module_wrapper_197582:
)
module_wrapper_1_197585:
%
module_wrapper_1_197587:)
module_wrapper_2_197590:
%
module_wrapper_2_197592:
)
module_wrapper_3_197595:
%
module_wrapper_3_197597:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¢(module_wrapper_2/StatefulPartitionedCall¢(module_wrapper_3/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_197580module_wrapper_197582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197295¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_197585module_wrapper_1_197587*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197312»
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_197590module_wrapper_2_197592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197329»
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_197595module_wrapper_3_197597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197345
IdentityIdentity1module_wrapper_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
Ò

1__inference_module_wrapper_1_layer_call_fn_197807

args_0
unknown:

	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
Ò

1__inference_module_wrapper_2_layer_call_fn_197847

args_0
unknown:

	unknown_0:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197329o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
×

L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197423

args_08
&ukryta2_matmul_readvariableop_resource:
5
'ukryta2_biasadd_readvariableop_resource:

identity¢ukryta2/BiasAdd/ReadVariableOp¢ukryta2/MatMul/ReadVariableOp
ukryta2/MatMul/ReadVariableOpReadVariableOp&ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta2/MatMulMatMulargs_0%ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta2/BiasAdd/ReadVariableOpReadVariableOp'ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta2/BiasAddBiasAddukryta2/MatMul:product:0&ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
ukryta2/ReluReluukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentityukryta2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^ukryta2/BiasAdd/ReadVariableOp^ukryta2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
ukryta2/BiasAdd/ReadVariableOpukryta2/BiasAdd/ReadVariableOp2>
ukryta2/MatMul/ReadVariableOpukryta2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
Ò

1__inference_module_wrapper_2_layer_call_fn_197856

args_0
unknown:

	unknown_0:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197423o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
ì	
È
+__inference_sequential_layer_call_fn_197371
module_wrapper_input
unknown:	

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_197352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
Â	
º
+__inference_sequential_layer_call_fn_197696

inputs
unknown:	

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_197537o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

¥
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197906

args_0:
(wyjsciowa_matmul_readvariableop_resource:
7
)wyjsciowa_biasadd_readvariableop_resource:
identity¢ wyjsciowa/BiasAdd/ReadVariableOp¢wyjsciowa/MatMul/ReadVariableOp
wyjsciowa/MatMul/ReadVariableOpReadVariableOp(wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
wyjsciowa/MatMulMatMulargs_0'wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp)wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
wyjsciowa/BiasAddBiasAddwyjsciowa/MatMul:product:0(wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitywyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^wyjsciowa/BiasAdd/ReadVariableOp ^wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2D
 wyjsciowa/BiasAdd/ReadVariableOp wyjsciowa/BiasAdd/ReadVariableOp2B
wyjsciowa/MatMul/ReadVariableOpwyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
ë

J__inference_module_wrapper_layer_call_and_return_conditional_losses_197483

args_09
'ukryta_1_matmul_readvariableop_resource:	
6
(ukryta_1_biasadd_readvariableop_resource:

identity¢ukryta_1/BiasAdd/ReadVariableOp¢ukryta_1/MatMul/ReadVariableOp
ukryta_1/MatMul/ReadVariableOpReadVariableOp'ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0{
ukryta_1/MatMulMatMulargs_0&ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta_1/BiasAdd/ReadVariableOpReadVariableOp(ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta_1/BiasAddBiasAddukryta_1/MatMul:product:0'ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
ukryta_1/ReluReluukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityukryta_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp ^ukryta_1/BiasAdd/ReadVariableOp^ukryta_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 2B
ukryta_1/BiasAdd/ReadVariableOpukryta_1/BiasAdd/ReadVariableOp2@
ukryta_1/MatMul/ReadVariableOpukryta_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0
×

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197453

args_08
&ukryta1_matmul_readvariableop_resource:
5
'ukryta1_biasadd_readvariableop_resource:
identity¢ukryta1/BiasAdd/ReadVariableOp¢ukryta1/MatMul/ReadVariableOp
ukryta1/MatMul/ReadVariableOpReadVariableOp&ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta1/MatMulMatMulargs_0%ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ukryta1/BiasAdd/ReadVariableOpReadVariableOp'ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ukryta1/BiasAddBiasAddukryta1/MatMul:product:0&ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
ukryta1/ReluReluukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityukryta1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ukryta1/BiasAdd/ReadVariableOp^ukryta1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2@
ukryta1/BiasAdd/ReadVariableOpukryta1/BiasAdd/ReadVariableOp2>
ukryta1/MatMul/ReadVariableOpukryta1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0

¥
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197393

args_0:
(wyjsciowa_matmul_readvariableop_resource:
7
)wyjsciowa_biasadd_readvariableop_resource:
identity¢ wyjsciowa/BiasAdd/ReadVariableOp¢wyjsciowa/MatMul/ReadVariableOp
wyjsciowa/MatMul/ReadVariableOpReadVariableOp(wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
wyjsciowa/MatMulMatMulargs_0'wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp)wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
wyjsciowa/BiasAddBiasAddwyjsciowa/MatMul:product:0(wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitywyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^wyjsciowa/BiasAdd/ReadVariableOp ^wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2D
 wyjsciowa/BiasAdd/ReadVariableOp wyjsciowa/BiasAdd/ReadVariableOp2B
wyjsciowa/MatMul/ReadVariableOpwyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
Î

/__inference_module_wrapper_layer_call_fn_197767

args_0
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0
À	
Á
$__inference_signature_wrapper_197654
module_wrapper_input
unknown:	

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_197277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
Ò

1__inference_module_wrapper_1_layer_call_fn_197816

args_0
unknown:

	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0

¥
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197345

args_0:
(wyjsciowa_matmul_readvariableop_resource:
7
)wyjsciowa_biasadd_readvariableop_resource:
identity¢ wyjsciowa/BiasAdd/ReadVariableOp¢wyjsciowa/MatMul/ReadVariableOp
wyjsciowa/MatMul/ReadVariableOpReadVariableOp(wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
wyjsciowa/MatMulMatMulargs_0'wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp)wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
wyjsciowa/BiasAddBiasAddwyjsciowa/MatMul:product:0(wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitywyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^wyjsciowa/BiasAdd/ReadVariableOp ^wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2D
 wyjsciowa/BiasAdd/ReadVariableOp wyjsciowa/BiasAdd/ReadVariableOp2B
wyjsciowa/MatMul/ReadVariableOpwyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
µ
Ó
F__inference_sequential_layer_call_and_return_conditional_losses_197537

inputs'
module_wrapper_197516:	
#
module_wrapper_197518:
)
module_wrapper_1_197521:
%
module_wrapper_1_197523:)
module_wrapper_2_197526:
%
module_wrapper_2_197528:
)
module_wrapper_3_197531:
%
module_wrapper_3_197533:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¢(module_wrapper_2/StatefulPartitionedCall¢(module_wrapper_3/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_197516module_wrapper_197518*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197483¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_197521module_wrapper_1_197523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197453»
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_197526module_wrapper_2_197528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197423»
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_197531module_wrapper_3_197533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197393
IdentityIdentity1module_wrapper_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ò

1__inference_module_wrapper_3_layer_call_fn_197896

args_0
unknown:

	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197393o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0

¥
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197916

args_0:
(wyjsciowa_matmul_readvariableop_resource:
7
)wyjsciowa_biasadd_readvariableop_resource:
identity¢ wyjsciowa/BiasAdd/ReadVariableOp¢wyjsciowa/MatMul/ReadVariableOp
wyjsciowa/MatMul/ReadVariableOpReadVariableOp(wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0}
wyjsciowa/MatMulMatMulargs_0'wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp)wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
wyjsciowa/BiasAddBiasAddwyjsciowa/MatMul:product:0(wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitywyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp!^wyjsciowa/BiasAdd/ReadVariableOp ^wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2D
 wyjsciowa/BiasAdd/ReadVariableOp wyjsciowa/BiasAdd/ReadVariableOp2B
wyjsciowa/MatMul/ReadVariableOpwyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
×

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197838

args_08
&ukryta1_matmul_readvariableop_resource:
5
'ukryta1_biasadd_readvariableop_resource:
identity¢ukryta1/BiasAdd/ReadVariableOp¢ukryta1/MatMul/ReadVariableOp
ukryta1/MatMul/ReadVariableOpReadVariableOp&ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta1/MatMulMatMulargs_0%ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ukryta1/BiasAdd/ReadVariableOpReadVariableOp'ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ukryta1/BiasAddBiasAddukryta1/MatMul:product:0&ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
ukryta1/ReluReluukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityukryta1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ukryta1/BiasAdd/ReadVariableOp^ukryta1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2@
ukryta1/BiasAdd/ReadVariableOpukryta1/BiasAdd/ReadVariableOp2>
ukryta1/MatMul/ReadVariableOpukryta1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
µ
Ê
"__inference__traced_restore_198159
file_prefixA
/assignvariableop_module_wrapper_ukryta_1_kernel:	
=
/assignvariableop_1_module_wrapper_ukryta_1_bias:
D
2assignvariableop_2_module_wrapper_1_ukryta1_kernel:
>
0assignvariableop_3_module_wrapper_1_ukryta1_bias:D
2assignvariableop_4_module_wrapper_2_ukryta2_kernel:
>
0assignvariableop_5_module_wrapper_2_ukryta2_bias:
F
4assignvariableop_6_module_wrapper_3_wyjsciowa_kernel:
@
2assignvariableop_7_module_wrapper_3_wyjsciowa_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: %
assignvariableop_13_total_2: %
assignvariableop_14_count_2: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: K
9assignvariableop_19_adam_module_wrapper_ukryta_1_kernel_m:	
E
7assignvariableop_20_adam_module_wrapper_ukryta_1_bias_m:
L
:assignvariableop_21_adam_module_wrapper_1_ukryta1_kernel_m:
F
8assignvariableop_22_adam_module_wrapper_1_ukryta1_bias_m:L
:assignvariableop_23_adam_module_wrapper_2_ukryta2_kernel_m:
F
8assignvariableop_24_adam_module_wrapper_2_ukryta2_bias_m:
N
<assignvariableop_25_adam_module_wrapper_3_wyjsciowa_kernel_m:
H
:assignvariableop_26_adam_module_wrapper_3_wyjsciowa_bias_m:K
9assignvariableop_27_adam_module_wrapper_ukryta_1_kernel_v:	
E
7assignvariableop_28_adam_module_wrapper_ukryta_1_bias_v:
L
:assignvariableop_29_adam_module_wrapper_1_ukryta1_kernel_v:
F
8assignvariableop_30_adam_module_wrapper_1_ukryta1_bias_v:L
:assignvariableop_31_adam_module_wrapper_2_ukryta2_kernel_v:
F
8assignvariableop_32_adam_module_wrapper_2_ukryta2_bias_v:
N
<assignvariableop_33_adam_module_wrapper_3_wyjsciowa_kernel_v:
H
:assignvariableop_34_adam_module_wrapper_3_wyjsciowa_bias_v:
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¶
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueÒBÏ$B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp/assignvariableop_module_wrapper_ukryta_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp/assignvariableop_1_module_wrapper_ukryta_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_2AssignVariableOp2assignvariableop_2_module_wrapper_1_ukryta1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp0assignvariableop_3_module_wrapper_1_ukryta1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_4AssignVariableOp2assignvariableop_4_module_wrapper_2_ukryta2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp0assignvariableop_5_module_wrapper_2_ukryta2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_6AssignVariableOp4assignvariableop_6_module_wrapper_3_wyjsciowa_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_7AssignVariableOp2assignvariableop_7_module_wrapper_3_wyjsciowa_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_19AssignVariableOp9assignvariableop_19_adam_module_wrapper_ukryta_1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_20AssignVariableOp7assignvariableop_20_adam_module_wrapper_ukryta_1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_21AssignVariableOp:assignvariableop_21_adam_module_wrapper_1_ukryta1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adam_module_wrapper_1_ukryta1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_23AssignVariableOp:assignvariableop_23_adam_module_wrapper_2_ukryta2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_24AssignVariableOp8assignvariableop_24_adam_module_wrapper_2_ukryta2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_25AssignVariableOp<assignvariableop_25_adam_module_wrapper_3_wyjsciowa_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_26AssignVariableOp:assignvariableop_26_adam_module_wrapper_3_wyjsciowa_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_27AssignVariableOp9assignvariableop_27_adam_module_wrapper_ukryta_1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_28AssignVariableOp7assignvariableop_28_adam_module_wrapper_ukryta_1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_29AssignVariableOp:assignvariableop_29_adam_module_wrapper_1_ukryta1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_30AssignVariableOp8assignvariableop_30_adam_module_wrapper_1_ukryta1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_module_wrapper_2_ukryta2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_module_wrapper_2_ukryta2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:­
AssignVariableOp_33AssignVariableOp<assignvariableop_33_adam_module_wrapper_3_wyjsciowa_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_34AssignVariableOp:assignvariableop_34_adam_module_wrapper_3_wyjsciowa_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ñ
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ¾
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
µ
Ó
F__inference_sequential_layer_call_and_return_conditional_losses_197352

inputs'
module_wrapper_197296:	
#
module_wrapper_197298:
)
module_wrapper_1_197313:
%
module_wrapper_1_197315:)
module_wrapper_2_197330:
%
module_wrapper_2_197332:
)
module_wrapper_3_197346:
%
module_wrapper_3_197348:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¢(module_wrapper_2/StatefulPartitionedCall¢(module_wrapper_3/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallinputsmodule_wrapper_197296module_wrapper_197298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197295¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_197313module_wrapper_1_197315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197312»
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_197330module_wrapper_2_197332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197329»
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_197346module_wrapper_3_197348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197345
IdentityIdentity1module_wrapper_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
×

L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197329

args_08
&ukryta2_matmul_readvariableop_resource:
5
'ukryta2_biasadd_readvariableop_resource:

identity¢ukryta2/BiasAdd/ReadVariableOp¢ukryta2/MatMul/ReadVariableOp
ukryta2/MatMul/ReadVariableOpReadVariableOp&ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta2/MatMulMatMulargs_0%ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta2/BiasAdd/ReadVariableOpReadVariableOp'ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta2/BiasAddBiasAddukryta2/MatMul:product:0&ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
ukryta2/ReluReluukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentityukryta2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^ukryta2/BiasAdd/ReadVariableOp^ukryta2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
ukryta2/BiasAdd/ReadVariableOpukryta2/BiasAdd/ReadVariableOp2>
ukryta2/MatMul/ReadVariableOpukryta2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
×

L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197878

args_08
&ukryta2_matmul_readvariableop_resource:
5
'ukryta2_biasadd_readvariableop_resource:

identity¢ukryta2/BiasAdd/ReadVariableOp¢ukryta2/MatMul/ReadVariableOp
ukryta2/MatMul/ReadVariableOpReadVariableOp&ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta2/MatMulMatMulargs_0%ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta2/BiasAdd/ReadVariableOpReadVariableOp'ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta2/BiasAddBiasAddukryta2/MatMul:product:0&ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
ukryta2/ReluReluukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentityukryta2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^ukryta2/BiasAdd/ReadVariableOp^ukryta2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
ukryta2/BiasAdd/ReadVariableOpukryta2/BiasAdd/ReadVariableOp2>
ukryta2/MatMul/ReadVariableOpukryta2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
ë

J__inference_module_wrapper_layer_call_and_return_conditional_losses_197787

args_09
'ukryta_1_matmul_readvariableop_resource:	
6
(ukryta_1_biasadd_readvariableop_resource:

identity¢ukryta_1/BiasAdd/ReadVariableOp¢ukryta_1/MatMul/ReadVariableOp
ukryta_1/MatMul/ReadVariableOpReadVariableOp'ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0{
ukryta_1/MatMulMatMulargs_0&ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta_1/BiasAdd/ReadVariableOpReadVariableOp(ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta_1/BiasAddBiasAddukryta_1/MatMul:product:0'ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
ukryta_1/ReluReluukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityukryta_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp ^ukryta_1/BiasAdd/ReadVariableOp^ukryta_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 2B
ukryta_1/BiasAdd/ReadVariableOpukryta_1/BiasAdd/ReadVariableOp2@
ukryta_1/MatMul/ReadVariableOpukryta_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0
×

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197827

args_08
&ukryta1_matmul_readvariableop_resource:
5
'ukryta1_biasadd_readvariableop_resource:
identity¢ukryta1/BiasAdd/ReadVariableOp¢ukryta1/MatMul/ReadVariableOp
ukryta1/MatMul/ReadVariableOpReadVariableOp&ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta1/MatMulMatMulargs_0%ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ukryta1/BiasAdd/ReadVariableOpReadVariableOp'ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ukryta1/BiasAddBiasAddukryta1/MatMul:product:0&ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
ukryta1/ReluReluukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityukryta1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ukryta1/BiasAdd/ReadVariableOp^ukryta1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2@
ukryta1/BiasAdd/ReadVariableOpukryta1/BiasAdd/ReadVariableOp2>
ukryta1/MatMul/ReadVariableOpukryta1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
×

L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197312

args_08
&ukryta1_matmul_readvariableop_resource:
5
'ukryta1_biasadd_readvariableop_resource:
identity¢ukryta1/BiasAdd/ReadVariableOp¢ukryta1/MatMul/ReadVariableOp
ukryta1/MatMul/ReadVariableOpReadVariableOp&ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta1/MatMulMatMulargs_0%ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ukryta1/BiasAdd/ReadVariableOpReadVariableOp'ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
ukryta1/BiasAddBiasAddukryta1/MatMul:product:0&ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
ukryta1/ReluReluukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityukryta1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^ukryta1/BiasAdd/ReadVariableOp^ukryta1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 2@
ukryta1/BiasAdd/ReadVariableOpukryta1/BiasAdd/ReadVariableOp2>
ukryta1/MatMul/ReadVariableOpukryta1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
×

L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197867

args_08
&ukryta2_matmul_readvariableop_resource:
5
'ukryta2_biasadd_readvariableop_resource:

identity¢ukryta2/BiasAdd/ReadVariableOp¢ukryta2/MatMul/ReadVariableOp
ukryta2/MatMul/ReadVariableOpReadVariableOp&ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0y
ukryta2/MatMulMatMulargs_0%ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta2/BiasAdd/ReadVariableOpReadVariableOp'ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta2/BiasAddBiasAddukryta2/MatMul:product:0&ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
ukryta2/ReluReluukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
i
IdentityIdentityukryta2/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp^ukryta2/BiasAdd/ReadVariableOp^ukryta2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2@
ukryta2/BiasAdd/ReadVariableOpukryta2/BiasAdd/ReadVariableOp2>
ukryta2/MatMul/ReadVariableOpukryta2/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0
ðK

__inference__traced_save_198044
file_prefix=
9savev2_module_wrapper_ukryta_1_kernel_read_readvariableop;
7savev2_module_wrapper_ukryta_1_bias_read_readvariableop>
:savev2_module_wrapper_1_ukryta1_kernel_read_readvariableop<
8savev2_module_wrapper_1_ukryta1_bias_read_readvariableop>
:savev2_module_wrapper_2_ukryta2_kernel_read_readvariableop<
8savev2_module_wrapper_2_ukryta2_bias_read_readvariableop@
<savev2_module_wrapper_3_wyjsciowa_kernel_read_readvariableop>
:savev2_module_wrapper_3_wyjsciowa_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_adam_module_wrapper_ukryta_1_kernel_m_read_readvariableopB
>savev2_adam_module_wrapper_ukryta_1_bias_m_read_readvariableopE
Asavev2_adam_module_wrapper_1_ukryta1_kernel_m_read_readvariableopC
?savev2_adam_module_wrapper_1_ukryta1_bias_m_read_readvariableopE
Asavev2_adam_module_wrapper_2_ukryta2_kernel_m_read_readvariableopC
?savev2_adam_module_wrapper_2_ukryta2_bias_m_read_readvariableopG
Csavev2_adam_module_wrapper_3_wyjsciowa_kernel_m_read_readvariableopE
Asavev2_adam_module_wrapper_3_wyjsciowa_bias_m_read_readvariableopD
@savev2_adam_module_wrapper_ukryta_1_kernel_v_read_readvariableopB
>savev2_adam_module_wrapper_ukryta_1_bias_v_read_readvariableopE
Asavev2_adam_module_wrapper_1_ukryta1_kernel_v_read_readvariableopC
?savev2_adam_module_wrapper_1_ukryta1_bias_v_read_readvariableopE
Asavev2_adam_module_wrapper_2_ukryta2_kernel_v_read_readvariableopC
?savev2_adam_module_wrapper_2_ukryta2_bias_v_read_readvariableopG
Csavev2_adam_module_wrapper_3_wyjsciowa_kernel_v_read_readvariableopE
Asavev2_adam_module_wrapper_3_wyjsciowa_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ³
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ü
valueÒBÏ$B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ñ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:09savev2_module_wrapper_ukryta_1_kernel_read_readvariableop7savev2_module_wrapper_ukryta_1_bias_read_readvariableop:savev2_module_wrapper_1_ukryta1_kernel_read_readvariableop8savev2_module_wrapper_1_ukryta1_bias_read_readvariableop:savev2_module_wrapper_2_ukryta2_kernel_read_readvariableop8savev2_module_wrapper_2_ukryta2_bias_read_readvariableop<savev2_module_wrapper_3_wyjsciowa_kernel_read_readvariableop:savev2_module_wrapper_3_wyjsciowa_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_adam_module_wrapper_ukryta_1_kernel_m_read_readvariableop>savev2_adam_module_wrapper_ukryta_1_bias_m_read_readvariableopAsavev2_adam_module_wrapper_1_ukryta1_kernel_m_read_readvariableop?savev2_adam_module_wrapper_1_ukryta1_bias_m_read_readvariableopAsavev2_adam_module_wrapper_2_ukryta2_kernel_m_read_readvariableop?savev2_adam_module_wrapper_2_ukryta2_bias_m_read_readvariableopCsavev2_adam_module_wrapper_3_wyjsciowa_kernel_m_read_readvariableopAsavev2_adam_module_wrapper_3_wyjsciowa_bias_m_read_readvariableop@savev2_adam_module_wrapper_ukryta_1_kernel_v_read_readvariableop>savev2_adam_module_wrapper_ukryta_1_bias_v_read_readvariableopAsavev2_adam_module_wrapper_1_ukryta1_kernel_v_read_readvariableop?savev2_adam_module_wrapper_1_ukryta1_bias_v_read_readvariableopAsavev2_adam_module_wrapper_2_ukryta2_kernel_v_read_readvariableop?savev2_adam_module_wrapper_2_ukryta2_bias_v_read_readvariableopCsavev2_adam_module_wrapper_3_wyjsciowa_kernel_v_read_readvariableopAsavev2_adam_module_wrapper_3_wyjsciowa_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ï
_input_shapesÝ
Ú: :	
:
:
::
:
:
:: : : : : : : : : : : :	
:
:
::
:
:
::	
:
:
::
:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:	
: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$  

_output_shapes

:
: !

_output_shapes
:
:$" 

_output_shapes

:
: #

_output_shapes
::$

_output_shapes
: 
6
Ú	
!__inference__wrapped_model_197277
module_wrapper_inputS
Asequential_module_wrapper_ukryta_1_matmul_readvariableop_resource:	
P
Bsequential_module_wrapper_ukryta_1_biasadd_readvariableop_resource:
T
Bsequential_module_wrapper_1_ukryta1_matmul_readvariableop_resource:
Q
Csequential_module_wrapper_1_ukryta1_biasadd_readvariableop_resource:T
Bsequential_module_wrapper_2_ukryta2_matmul_readvariableop_resource:
Q
Csequential_module_wrapper_2_ukryta2_biasadd_readvariableop_resource:
V
Dsequential_module_wrapper_3_wyjsciowa_matmul_readvariableop_resource:
S
Esequential_module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource:
identity¢9sequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOp¢8sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOp¢:sequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp¢9sequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOp¢:sequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp¢9sequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOp¢<sequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp¢;sequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOpº
8sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOpReadVariableOpAsequential_module_wrapper_ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0½
)sequential/module_wrapper/ukryta_1/MatMulMatMulmodule_wrapper_input@sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¸
9sequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOpReadVariableOpBsequential_module_wrapper_ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0ß
*sequential/module_wrapper/ukryta_1/BiasAddBiasAdd3sequential/module_wrapper/ukryta_1/MatMul:product:0Asequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

'sequential/module_wrapper/ukryta_1/ReluRelu3sequential/module_wrapper/ukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
9sequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOpReadVariableOpBsequential_module_wrapper_1_ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0à
*sequential/module_wrapper_1/ukryta1/MatMulMatMul5sequential/module_wrapper/ukryta_1/Relu:activations:0Asequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
:sequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOpReadVariableOpCsequential_module_wrapper_1_ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0â
+sequential/module_wrapper_1/ukryta1/BiasAddBiasAdd4sequential/module_wrapper_1/ukryta1/MatMul:product:0Bsequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential/module_wrapper_1/ukryta1/ReluRelu4sequential/module_wrapper_1/ukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOpReadVariableOpBsequential_module_wrapper_2_ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0á
*sequential/module_wrapper_2/ukryta2/MatMulMatMul6sequential/module_wrapper_1/ukryta1/Relu:activations:0Asequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
:sequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOpReadVariableOpCsequential_module_wrapper_2_ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0â
+sequential/module_wrapper_2/ukryta2/BiasAddBiasAdd4sequential/module_wrapper_2/ukryta2/MatMul:product:0Bsequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

(sequential/module_wrapper_2/ukryta2/ReluRelu4sequential/module_wrapper_2/ukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
À
;sequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOpReadVariableOpDsequential_module_wrapper_3_wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0å
,sequential/module_wrapper_3/wyjsciowa/MatMulMatMul6sequential/module_wrapper_2/ukryta2/Relu:activations:0Csequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
<sequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOpReadVariableOpEsequential_module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0è
-sequential/module_wrapper_3/wyjsciowa/BiasAddBiasAdd6sequential/module_wrapper_3/wyjsciowa/MatMul:product:0Dsequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity6sequential/module_wrapper_3/wyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
NoOpNoOp:^sequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOp9^sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOp;^sequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp:^sequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOp;^sequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp:^sequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOp=^sequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp<^sequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2v
9sequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOp9sequential/module_wrapper/ukryta_1/BiasAdd/ReadVariableOp2t
8sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOp8sequential/module_wrapper/ukryta_1/MatMul/ReadVariableOp2x
:sequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp:sequential/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp2v
9sequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOp9sequential/module_wrapper_1/ukryta1/MatMul/ReadVariableOp2x
:sequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp:sequential/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp2v
9sequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOp9sequential/module_wrapper_2/ukryta2/MatMul/ReadVariableOp2|
<sequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp<sequential/module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp2z
;sequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp;sequential/module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
Ò

1__inference_module_wrapper_3_layer_call_fn_197887

args_0
unknown:

	unknown_0:
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197345o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameargs_0
Â	
º
+__inference_sequential_layer_call_fn_197675

inputs
unknown:	

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_197352o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ì	
È
+__inference_sequential_layer_call_fn_197577
module_wrapper_input
unknown:	

	unknown_0:

	unknown_1:

	unknown_2:
	unknown_3:

	unknown_4:

	unknown_5:

	unknown_6:
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_197537o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
Î

/__inference_module_wrapper_layer_call_fn_197776

args_0
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197483o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0
Ñ.
Á
F__inference_sequential_layer_call_and_return_conditional_losses_197758

inputsH
6module_wrapper_ukryta_1_matmul_readvariableop_resource:	
E
7module_wrapper_ukryta_1_biasadd_readvariableop_resource:
I
7module_wrapper_1_ukryta1_matmul_readvariableop_resource:
F
8module_wrapper_1_ukryta1_biasadd_readvariableop_resource:I
7module_wrapper_2_ukryta2_matmul_readvariableop_resource:
F
8module_wrapper_2_ukryta2_biasadd_readvariableop_resource:
K
9module_wrapper_3_wyjsciowa_matmul_readvariableop_resource:
H
:module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource:
identity¢.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp¢-module_wrapper/ukryta_1/MatMul/ReadVariableOp¢/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp¢.module_wrapper_1/ukryta1/MatMul/ReadVariableOp¢/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp¢.module_wrapper_2/ukryta2/MatMul/ReadVariableOp¢1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp¢0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp¤
-module_wrapper/ukryta_1/MatMul/ReadVariableOpReadVariableOp6module_wrapper_ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0
module_wrapper/ukryta_1/MatMulMatMulinputs5module_wrapper/ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
.module_wrapper/ukryta_1/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¾
module_wrapper/ukryta_1/BiasAddBiasAdd(module_wrapper/ukryta_1/MatMul:product:06module_wrapper/ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

module_wrapper/ukryta_1/ReluRelu(module_wrapper/ukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
.module_wrapper_1/ukryta1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_1_ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¿
module_wrapper_1/ukryta1/MatMulMatMul*module_wrapper/ukryta_1/Relu:activations:06module_wrapper_1/ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_1_ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 module_wrapper_1/ukryta1/BiasAddBiasAdd)module_wrapper_1/ukryta1/MatMul:product:07module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
module_wrapper_1/ukryta1/ReluRelu)module_wrapper_1/ukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper_2/ukryta2/MatMul/ReadVariableOpReadVariableOp7module_wrapper_2_ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0À
module_wrapper_2/ukryta2/MatMulMatMul+module_wrapper_1/ukryta1/Relu:activations:06module_wrapper_2/ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_2_ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Á
 module_wrapper_2/ukryta2/BiasAddBiasAdd)module_wrapper_2/ukryta2/MatMul:product:07module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

module_wrapper_2/ukryta2/ReluRelu)module_wrapper_2/ukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOpReadVariableOp9module_wrapper_3_wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ä
!module_wrapper_3/wyjsciowa/MatMulMatMul+module_wrapper_2/ukryta2/Relu:activations:08module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
"module_wrapper_3/wyjsciowa/BiasAddBiasAdd+module_wrapper_3/wyjsciowa/MatMul:product:09module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+module_wrapper_3/wyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp/^module_wrapper/ukryta_1/BiasAdd/ReadVariableOp.^module_wrapper/ukryta_1/MatMul/ReadVariableOp0^module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp/^module_wrapper_1/ukryta1/MatMul/ReadVariableOp0^module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp/^module_wrapper_2/ukryta2/MatMul/ReadVariableOp2^module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp1^module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2`
.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp2^
-module_wrapper/ukryta_1/MatMul/ReadVariableOp-module_wrapper/ukryta_1/MatMul/ReadVariableOp2b
/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp2`
.module_wrapper_1/ukryta1/MatMul/ReadVariableOp.module_wrapper_1/ukryta1/MatMul/ReadVariableOp2b
/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp2`
.module_wrapper_2/ukryta2/MatMul/ReadVariableOp.module_wrapper_2/ukryta2/MatMul/ReadVariableOp2f
1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp2d
0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ñ.
Á
F__inference_sequential_layer_call_and_return_conditional_losses_197727

inputsH
6module_wrapper_ukryta_1_matmul_readvariableop_resource:	
E
7module_wrapper_ukryta_1_biasadd_readvariableop_resource:
I
7module_wrapper_1_ukryta1_matmul_readvariableop_resource:
F
8module_wrapper_1_ukryta1_biasadd_readvariableop_resource:I
7module_wrapper_2_ukryta2_matmul_readvariableop_resource:
F
8module_wrapper_2_ukryta2_biasadd_readvariableop_resource:
K
9module_wrapper_3_wyjsciowa_matmul_readvariableop_resource:
H
:module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource:
identity¢.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp¢-module_wrapper/ukryta_1/MatMul/ReadVariableOp¢/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp¢.module_wrapper_1/ukryta1/MatMul/ReadVariableOp¢/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp¢.module_wrapper_2/ukryta2/MatMul/ReadVariableOp¢1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp¢0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp¤
-module_wrapper/ukryta_1/MatMul/ReadVariableOpReadVariableOp6module_wrapper_ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0
module_wrapper/ukryta_1/MatMulMatMulinputs5module_wrapper/ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¢
.module_wrapper/ukryta_1/BiasAdd/ReadVariableOpReadVariableOp7module_wrapper_ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0¾
module_wrapper/ukryta_1/BiasAddBiasAdd(module_wrapper/ukryta_1/MatMul:product:06module_wrapper/ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

module_wrapper/ukryta_1/ReluRelu(module_wrapper/ukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¦
.module_wrapper_1/ukryta1/MatMul/ReadVariableOpReadVariableOp7module_wrapper_1_ukryta1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0¿
module_wrapper_1/ukryta1/MatMulMatMul*module_wrapper/ukryta_1/Relu:activations:06module_wrapper_1/ukryta1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_1_ukryta1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
 module_wrapper_1/ukryta1/BiasAddBiasAdd)module_wrapper_1/ukryta1/MatMul:product:07module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
module_wrapper_1/ukryta1/ReluRelu)module_wrapper_1/ukryta1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
.module_wrapper_2/ukryta2/MatMul/ReadVariableOpReadVariableOp7module_wrapper_2_ukryta2_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0À
module_wrapper_2/ukryta2/MatMulMatMul+module_wrapper_1/ukryta1/Relu:activations:06module_wrapper_2/ukryta2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¤
/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOpReadVariableOp8module_wrapper_2_ukryta2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Á
 module_wrapper_2/ukryta2/BiasAddBiasAdd)module_wrapper_2/ukryta2/MatMul:product:07module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

module_wrapper_2/ukryta2/ReluRelu)module_wrapper_2/ukryta2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ª
0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOpReadVariableOp9module_wrapper_3_wyjsciowa_matmul_readvariableop_resource*
_output_shapes

:
*
dtype0Ä
!module_wrapper_3/wyjsciowa/MatMulMatMul+module_wrapper_2/ukryta2/Relu:activations:08module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOpReadVariableOp:module_wrapper_3_wyjsciowa_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ç
"module_wrapper_3/wyjsciowa/BiasAddBiasAdd+module_wrapper_3/wyjsciowa/MatMul:product:09module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+module_wrapper_3/wyjsciowa/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp/^module_wrapper/ukryta_1/BiasAdd/ReadVariableOp.^module_wrapper/ukryta_1/MatMul/ReadVariableOp0^module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp/^module_wrapper_1/ukryta1/MatMul/ReadVariableOp0^module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp/^module_wrapper_2/ukryta2/MatMul/ReadVariableOp2^module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp1^module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2`
.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp.module_wrapper/ukryta_1/BiasAdd/ReadVariableOp2^
-module_wrapper/ukryta_1/MatMul/ReadVariableOp-module_wrapper/ukryta_1/MatMul/ReadVariableOp2b
/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp/module_wrapper_1/ukryta1/BiasAdd/ReadVariableOp2`
.module_wrapper_1/ukryta1/MatMul/ReadVariableOp.module_wrapper_1/ukryta1/MatMul/ReadVariableOp2b
/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp/module_wrapper_2/ukryta2/BiasAdd/ReadVariableOp2`
.module_wrapper_2/ukryta2/MatMul/ReadVariableOp.module_wrapper_2/ukryta2/MatMul/ReadVariableOp2f
1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp1module_wrapper_3/wyjsciowa/BiasAdd/ReadVariableOp2d
0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp0module_wrapper_3/wyjsciowa/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
ß
á
F__inference_sequential_layer_call_and_return_conditional_losses_197625
module_wrapper_input'
module_wrapper_197604:	
#
module_wrapper_197606:
)
module_wrapper_1_197609:
%
module_wrapper_1_197611:)
module_wrapper_2_197614:
%
module_wrapper_2_197616:
)
module_wrapper_3_197619:
%
module_wrapper_3_197621:
identity¢&module_wrapper/StatefulPartitionedCall¢(module_wrapper_1/StatefulPartitionedCall¢(module_wrapper_2/StatefulPartitionedCall¢(module_wrapper_3/StatefulPartitionedCall
&module_wrapper/StatefulPartitionedCallStatefulPartitionedCallmodule_wrapper_inputmodule_wrapper_197604module_wrapper_197606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197483¹
(module_wrapper_1/StatefulPartitionedCallStatefulPartitionedCall/module_wrapper/StatefulPartitionedCall:output:0module_wrapper_1_197609module_wrapper_1_197611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197453»
(module_wrapper_2/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_1/StatefulPartitionedCall:output:0module_wrapper_2_197614module_wrapper_2_197616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197423»
(module_wrapper_3/StatefulPartitionedCallStatefulPartitionedCall1module_wrapper_2/StatefulPartitionedCall:output:0module_wrapper_3_197619module_wrapper_3_197621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197393
IdentityIdentity1module_wrapper_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp'^module_wrapper/StatefulPartitionedCall)^module_wrapper_1/StatefulPartitionedCall)^module_wrapper_2/StatefulPartitionedCall)^module_wrapper_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ	: : : : : : : : 2P
&module_wrapper/StatefulPartitionedCall&module_wrapper/StatefulPartitionedCall2T
(module_wrapper_1/StatefulPartitionedCall(module_wrapper_1/StatefulPartitionedCall2T
(module_wrapper_2/StatefulPartitionedCall(module_wrapper_2/StatefulPartitionedCall2T
(module_wrapper_3/StatefulPartitionedCall(module_wrapper_3/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
.
_user_specified_namemodule_wrapper_input
ë

J__inference_module_wrapper_layer_call_and_return_conditional_losses_197798

args_09
'ukryta_1_matmul_readvariableop_resource:	
6
(ukryta_1_biasadd_readvariableop_resource:

identity¢ukryta_1/BiasAdd/ReadVariableOp¢ukryta_1/MatMul/ReadVariableOp
ukryta_1/MatMul/ReadVariableOpReadVariableOp'ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0{
ukryta_1/MatMulMatMulargs_0&ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta_1/BiasAdd/ReadVariableOpReadVariableOp(ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta_1/BiasAddBiasAddukryta_1/MatMul:product:0'ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
ukryta_1/ReluReluukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityukryta_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp ^ukryta_1/BiasAdd/ReadVariableOp^ukryta_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 2B
ukryta_1/BiasAdd/ReadVariableOpukryta_1/BiasAdd/ReadVariableOp2@
ukryta_1/MatMul/ReadVariableOpukryta_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0
ë

J__inference_module_wrapper_layer_call_and_return_conditional_losses_197295

args_09
'ukryta_1_matmul_readvariableop_resource:	
6
(ukryta_1_biasadd_readvariableop_resource:

identity¢ukryta_1/BiasAdd/ReadVariableOp¢ukryta_1/MatMul/ReadVariableOp
ukryta_1/MatMul/ReadVariableOpReadVariableOp'ukryta_1_matmul_readvariableop_resource*
_output_shapes

:	
*
dtype0{
ukryta_1/MatMulMatMulargs_0&ukryta_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ukryta_1/BiasAdd/ReadVariableOpReadVariableOp(ukryta_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
ukryta_1/BiasAddBiasAddukryta_1/MatMul:product:0'ukryta_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
b
ukryta_1/ReluReluukryta_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityukryta_1/Relu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp ^ukryta_1/BiasAdd/ReadVariableOp^ukryta_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ	: : 2B
ukryta_1/BiasAdd/ReadVariableOpukryta_1/BiasAdd/ReadVariableOp2@
ukryta_1/MatMul/ReadVariableOpukryta_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameargs_0"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
U
module_wrapper_input=
&serving_default_module_wrapper_input:0ÿÿÿÿÿÿÿÿÿ	D
module_wrapper_30
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¶Ò

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
²
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
²
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_module"
_tf_keras_layer
²
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_module"
_tf_keras_layer
²
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_module"
_tf_keras_layer
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
â
7trace_0
8trace_1
9trace_2
:trace_32÷
+__inference_sequential_layer_call_fn_197371
+__inference_sequential_layer_call_fn_197675
+__inference_sequential_layer_call_fn_197696
+__inference_sequential_layer_call_fn_197577À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z7trace_0z8trace_1z9trace_2z:trace_3
Î
;trace_0
<trace_1
=trace_2
>trace_32ã
F__inference_sequential_layer_call_and_return_conditional_losses_197727
F__inference_sequential_layer_call_and_return_conditional_losses_197758
F__inference_sequential_layer_call_and_return_conditional_losses_197601
F__inference_sequential_layer_call_and_return_conditional_losses_197625À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z;trace_0z<trace_1z=trace_2z>trace_3
ÙBÖ
!__inference__wrapped_model_197277module_wrapper_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rate*m¦+m§,m¨-m©.mª/m«0m¬1m­*v®+v¯,v°-v±.v²/v³0v´1vµ"
	optimizer
,
Dserving_default"
signature_map
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ü
Jtrace_0
Ktrace_12¥
/__inference_module_wrapper_layer_call_fn_197767
/__inference_module_wrapper_layer_call_fn_197776À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zJtrace_0zKtrace_1

Ltrace_0
Mtrace_12Û
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197787
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197798À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zLtrace_0zMtrace_1
»
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
à
Ytrace_0
Ztrace_12©
1__inference_module_wrapper_1_layer_call_fn_197807
1__inference_module_wrapper_1_layer_call_fn_197816À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zYtrace_0zZtrace_1

[trace_0
\trace_12ß
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197827
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197838À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 z[trace_0z\trace_1
»
]trainable_variables
^	variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
à
htrace_0
itrace_12©
1__inference_module_wrapper_2_layer_call_fn_197847
1__inference_module_wrapper_2_layer_call_fn_197856À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zhtrace_0zitrace_1

jtrace_0
ktrace_12ß
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197867
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197878À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zjtrace_0zktrace_1
»
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

.kernel
/bias"
_tf_keras_layer
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
­
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
à
wtrace_0
xtrace_12©
1__inference_module_wrapper_3_layer_call_fn_197887
1__inference_module_wrapper_3_layer_call_fn_197896À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zwtrace_0zxtrace_1

ytrace_0
ztrace_12ß
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197906
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197916À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 zytrace_0zztrace_1
¼
{trainable_variables
|	variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

0kernel
1bias"
_tf_keras_layer
0:.	
2module_wrapper/ukryta_1/kernel
*:(
2module_wrapper/ukryta_1/bias
1:/
2module_wrapper_1/ukryta1/kernel
+:)2module_wrapper_1/ukryta1/bias
1:/
2module_wrapper_2/ukryta2/kernel
+:)
2module_wrapper_2/ukryta2/bias
3:1
2!module_wrapper_3/wyjsciowa/kernel
-:+2module_wrapper_3/wyjsciowa/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
+__inference_sequential_layer_call_fn_197371module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ýBú
+__inference_sequential_layer_call_fn_197675inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ýBú
+__inference_sequential_layer_call_fn_197696inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
+__inference_sequential_layer_call_fn_197577module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_197727inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
F__inference_sequential_layer_call_and_return_conditional_losses_197758inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦B£
F__inference_sequential_layer_call_and_return_conditional_losses_197601module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦B£
F__inference_sequential_layer_call_and_return_conditional_losses_197625module_wrapper_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ØBÕ
$__inference_signature_wrapper_197654module_wrapper_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bþ
/__inference_module_wrapper_layer_call_fn_197767args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Bþ
/__inference_module_wrapper_layer_call_fn_197776args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197787args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197798args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
layers
metrics
Ntrainable_variables
O	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
1__inference_module_wrapper_1_layer_call_fn_197807args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
1__inference_module_wrapper_1_layer_call_fn_197816args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197827args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197838args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
layers
metrics
]trainable_variables
^	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
1__inference_module_wrapper_2_layer_call_fn_197847args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
1__inference_module_wrapper_2_layer_call_fn_197856args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197867args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197878args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
layers
metrics
ltrainable_variables
m	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
1__inference_module_wrapper_3_layer_call_fn_197887args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
1__inference_module_wrapper_3_layer_call_fn_197896args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197906args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
B
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197916args_0"À
·²³
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
´
layers
metrics
{trainable_variables
|	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count
 
_fn_kwargs"
_tf_keras_metric
c
¡	variables
¢	keras_api

£total

¤count
¥
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
£0
¤1"
trackable_list_wrapper
.
¡	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
5:3	
2%Adam/module_wrapper/ukryta_1/kernel/m
/:-
2#Adam/module_wrapper/ukryta_1/bias/m
6:4
2&Adam/module_wrapper_1/ukryta1/kernel/m
0:.2$Adam/module_wrapper_1/ukryta1/bias/m
6:4
2&Adam/module_wrapper_2/ukryta2/kernel/m
0:.
2$Adam/module_wrapper_2/ukryta2/bias/m
8:6
2(Adam/module_wrapper_3/wyjsciowa/kernel/m
2:02&Adam/module_wrapper_3/wyjsciowa/bias/m
5:3	
2%Adam/module_wrapper/ukryta_1/kernel/v
/:-
2#Adam/module_wrapper/ukryta_1/bias/v
6:4
2&Adam/module_wrapper_1/ukryta1/kernel/v
0:.2$Adam/module_wrapper_1/ukryta1/bias/v
6:4
2&Adam/module_wrapper_2/ukryta2/kernel/v
0:.
2$Adam/module_wrapper_2/ukryta2/bias/v
8:6
2(Adam/module_wrapper_3/wyjsciowa/kernel/v
2:02&Adam/module_wrapper_3/wyjsciowa/bias/v´
!__inference__wrapped_model_197277*+,-./01=¢:
3¢0
.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	
ª "Cª@
>
module_wrapper_3*'
module_wrapper_3ÿÿÿÿÿÿÿÿÿ¼
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197827l,-?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
L__inference_module_wrapper_1_layer_call_and_return_conditional_losses_197838l,-?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_module_wrapper_1_layer_call_fn_197807_,-?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp "ÿÿÿÿÿÿÿÿÿ
1__inference_module_wrapper_1_layer_call_fn_197816_,-?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp"ÿÿÿÿÿÿÿÿÿ¼
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197867l./?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¼
L__inference_module_wrapper_2_layer_call_and_return_conditional_losses_197878l./?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ

 
1__inference_module_wrapper_2_layer_call_fn_197847_./?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp "ÿÿÿÿÿÿÿÿÿ

1__inference_module_wrapper_2_layer_call_fn_197856_./?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ
ª

trainingp"ÿÿÿÿÿÿÿÿÿ
¼
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197906l01?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
L__inference_module_wrapper_3_layer_call_and_return_conditional_losses_197916l01?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_module_wrapper_3_layer_call_fn_197887_01?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp "ÿÿÿÿÿÿÿÿÿ
1__inference_module_wrapper_3_layer_call_fn_197896_01?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ

ª

trainingp"ÿÿÿÿÿÿÿÿÿº
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197787l*+?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ	
ª

trainingp "%¢"

0ÿÿÿÿÿÿÿÿÿ

 º
J__inference_module_wrapper_layer_call_and_return_conditional_losses_197798l*+?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ	
ª

trainingp"%¢"

0ÿÿÿÿÿÿÿÿÿ

 
/__inference_module_wrapper_layer_call_fn_197767_*+?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ	
ª

trainingp "ÿÿÿÿÿÿÿÿÿ

/__inference_module_wrapper_layer_call_fn_197776_*+?¢<
%¢"
 
args_0ÿÿÿÿÿÿÿÿÿ	
ª

trainingp"ÿÿÿÿÿÿÿÿÿ
Â
F__inference_sequential_layer_call_and_return_conditional_losses_197601x*+,-./01E¢B
;¢8
.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
F__inference_sequential_layer_call_and_return_conditional_losses_197625x*+,-./01E¢B
;¢8
.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
F__inference_sequential_layer_call_and_return_conditional_losses_197727j*+,-./017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ	
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
F__inference_sequential_layer_call_and_return_conditional_losses_197758j*+,-./017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ	
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_197371k*+,-./01E¢B
;¢8
.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_197577k*+,-./01E¢B
;¢8
.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	
p

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_197675]*+,-./017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ	
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_197696]*+,-./017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ	
p

 
ª "ÿÿÿÿÿÿÿÿÿÏ
$__inference_signature_wrapper_197654¦*+,-./01U¢R
¢ 
KªH
F
module_wrapper_input.+
module_wrapper_inputÿÿÿÿÿÿÿÿÿ	"Cª@
>
module_wrapper_3*'
module_wrapper_3ÿÿÿÿÿÿÿÿÿ