??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
delete_old_dirsbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??	
?
,seq_seq_single_sations_cnn_1/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,seq_seq_single_sations_cnn_1/conv1d_1/kernel
?
@seq_seq_single_sations_cnn_1/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp,seq_seq_single_sations_cnn_1/conv1d_1/kernel*"
_output_shapes
:@*
dtype0
?
*seq_seq_single_sations_cnn_1/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*seq_seq_single_sations_cnn_1/conv1d_1/bias
?
>seq_seq_single_sations_cnn_1/conv1d_1/bias/Read/ReadVariableOpReadVariableOp*seq_seq_single_sations_cnn_1/conv1d_1/bias*
_output_shapes
:@*
dtype0
?
+seq_seq_single_sations_cnn_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *<
shared_name-+seq_seq_single_sations_cnn_1/dense_2/kernel
?
?seq_seq_single_sations_cnn_1/dense_2/kernel/Read/ReadVariableOpReadVariableOp+seq_seq_single_sations_cnn_1/dense_2/kernel*
_output_shapes

:@ *
dtype0
?
)seq_seq_single_sations_cnn_1/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)seq_seq_single_sations_cnn_1/dense_2/bias
?
=seq_seq_single_sations_cnn_1/dense_2/bias/Read/ReadVariableOpReadVariableOp)seq_seq_single_sations_cnn_1/dense_2/bias*
_output_shapes
: *
dtype0
?
+seq_seq_single_sations_cnn_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *<
shared_name-+seq_seq_single_sations_cnn_1/dense_3/kernel
?
?seq_seq_single_sations_cnn_1/dense_3/kernel/Read/ReadVariableOpReadVariableOp+seq_seq_single_sations_cnn_1/dense_3/kernel*
_output_shapes

: *
dtype0
?
)seq_seq_single_sations_cnn_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)seq_seq_single_sations_cnn_1/dense_3/bias
?
=seq_seq_single_sations_cnn_1/dense_3/bias/Read/ReadVariableOpReadVariableOp)seq_seq_single_sations_cnn_1/dense_3/bias*
_output_shapes
:*
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
?
3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m
?
GAdam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOp3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m*"
_output_shapes
:@*
dtype0
?
1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m
?
EAdam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOp1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m*
_output_shapes
:@*
dtype0
?
2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *C
shared_name42Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/m
?
FAdam/seq_seq_single_sations_cnn_1/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/m*
_output_shapes

:@ *
dtype0
?
0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/seq_seq_single_sations_cnn_1/dense_2/bias/m
?
DAdam/seq_seq_single_sations_cnn_1/dense_2/bias/m/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/m*
_output_shapes
: *
dtype0
?
2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/m
?
FAdam/seq_seq_single_sations_cnn_1/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/m*
_output_shapes

: *
dtype0
?
0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/seq_seq_single_sations_cnn_1/dense_3/bias/m
?
DAdam/seq_seq_single_sations_cnn_1/dense_3/bias/m/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/m*
_output_shapes
:*
dtype0
?
3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v
?
GAdam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOp3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v*"
_output_shapes
:@*
dtype0
?
1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v
?
EAdam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOp1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v*
_output_shapes
:@*
dtype0
?
2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *C
shared_name42Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/v
?
FAdam/seq_seq_single_sations_cnn_1/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/v*
_output_shapes

:@ *
dtype0
?
0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adam/seq_seq_single_sations_cnn_1/dense_2/bias/v
?
DAdam/seq_seq_single_sations_cnn_1/dense_2/bias/v/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/v*
_output_shapes
: *
dtype0
?
2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *C
shared_name42Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/v
?
FAdam/seq_seq_single_sations_cnn_1/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/v*
_output_shapes

: *
dtype0
?
0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/seq_seq_single_sations_cnn_1/dense_3/bias/v
?
DAdam/seq_seq_single_sations_cnn_1/dense_3/bias/v/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
cnn
batch_norm_layer
hidden_dense
	out_dense
conv_dropout
dense_dropout
reshape_output
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
R
!	variables
"regularization_losses
#trainable_variables
$	keras_api
R
%	variables
&regularization_losses
'trainable_variables
(	keras_api
R
)	variables
*regularization_losses
+trainable_variables
,	keras_api
?
-iter

.beta_1

/beta_2
	0decay
1learning_ratemZm[m\m]m^m_v`vavbvcvdve
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
?
2metrics
		variables
3non_trainable_variables

regularization_losses

4layers
5layer_regularization_losses
trainable_variables
6layer_metrics
 
ge
VARIABLE_VALUE,seq_seq_single_sations_cnn_1/conv1d_1/kernel%cnn/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE*seq_seq_single_sations_cnn_1/conv1d_1/bias#cnn/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
7metrics
	variables
8non_trainable_variables
regularization_losses

9layers
:layer_regularization_losses
trainable_variables
;layer_metrics
 
om
VARIABLE_VALUE+seq_seq_single_sations_cnn_1/dense_2/kernel.hidden_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE)seq_seq_single_sations_cnn_1/dense_2/bias,hidden_dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
<metrics
	variables
=non_trainable_variables
regularization_losses

>layers
?layer_regularization_losses
trainable_variables
@layer_metrics
lj
VARIABLE_VALUE+seq_seq_single_sations_cnn_1/dense_3/kernel+out_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE)seq_seq_single_sations_cnn_1/dense_3/bias)out_dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Ametrics
	variables
Bnon_trainable_variables
regularization_losses

Clayers
Dlayer_regularization_losses
trainable_variables
Elayer_metrics
 
 
 
?
Fmetrics
!	variables
Gnon_trainable_variables
"regularization_losses

Hlayers
Ilayer_regularization_losses
#trainable_variables
Jlayer_metrics
 
 
 
?
Kmetrics
%	variables
Lnon_trainable_variables
&regularization_losses

Mlayers
Nlayer_regularization_losses
'trainable_variables
Olayer_metrics
 
 
 
?
Pmetrics
)	variables
Qnon_trainable_variables
*regularization_losses

Rlayers
Slayer_regularization_losses
+trainable_variables
Tlayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

U0
 
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Vtotal
	Wcount
X	variables
Y	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

X	variables
??
VARIABLE_VALUE3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/mAcnn/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m?cnn/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/mJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/mHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/mGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/mEout_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/vAcnn/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v?cnn/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/vJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/vHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/vGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/vEout_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????+*
dtype0* 
shape:?????????+
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1,seq_seq_single_sations_cnn_1/conv1d_1/kernel*seq_seq_single_sations_cnn_1/conv1d_1/bias+seq_seq_single_sations_cnn_1/dense_2/kernel)seq_seq_single_sations_cnn_1/dense_2/bias+seq_seq_single_sations_cnn_1/dense_3/kernel)seq_seq_single_sations_cnn_1/dense_3/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_16114
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename@seq_seq_single_sations_cnn_1/conv1d_1/kernel/Read/ReadVariableOp>seq_seq_single_sations_cnn_1/conv1d_1/bias/Read/ReadVariableOp?seq_seq_single_sations_cnn_1/dense_2/kernel/Read/ReadVariableOp=seq_seq_single_sations_cnn_1/dense_2/bias/Read/ReadVariableOp?seq_seq_single_sations_cnn_1/dense_3/kernel/Read/ReadVariableOp=seq_seq_single_sations_cnn_1/dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpGAdam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m/Read/ReadVariableOpEAdam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m/Read/ReadVariableOpFAdam/seq_seq_single_sations_cnn_1/dense_2/kernel/m/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn_1/dense_2/bias/m/Read/ReadVariableOpFAdam/seq_seq_single_sations_cnn_1/dense_3/kernel/m/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn_1/dense_3/bias/m/Read/ReadVariableOpGAdam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v/Read/ReadVariableOpEAdam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v/Read/ReadVariableOpFAdam/seq_seq_single_sations_cnn_1/dense_2/kernel/v/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn_1/dense_2/bias/v/Read/ReadVariableOpFAdam/seq_seq_single_sations_cnn_1/dense_3/kernel/v/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn_1/dense_3/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_16583
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename,seq_seq_single_sations_cnn_1/conv1d_1/kernel*seq_seq_single_sations_cnn_1/conv1d_1/bias+seq_seq_single_sations_cnn_1/dense_2/kernel)seq_seq_single_sations_cnn_1/dense_2/bias+seq_seq_single_sations_cnn_1/dense_3/kernel)seq_seq_single_sations_cnn_1/dense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/m0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/m2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/m0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/m3Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v1Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v2Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/v0Adam/seq_seq_single_sations_cnn_1/dense_2/bias/v2Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/v0Adam/seq_seq_single_sations_cnn_1/dense_3/bias/v*%
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_16668??
?
E
)__inference_dropout_3_layer_call_fn_16463

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_159982
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_16114
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_157402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_16458

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_15876
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_158612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
?
?
(__inference_conv1d_1_layer_call_fn_16343

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_157672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
B__inference_dense_2_layer_call_and_return_conditional_losses_16374

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_16014

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
? 
?
B__inference_dense_3_layer_call_and_return_conditional_losses_16413

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16301
x
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_158612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????+

_user_specified_namex
?j
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16195
xJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_1_biasadd_readvariableop_resource:@;
)dense_2_tensordot_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: ;
)dense_3_tensordot_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsstrided_slice:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Relu?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free}
dense_2/Tensordot/ShapeShapeconv1d_1/Relu:activations:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposeconv1d_1/Relu:activations:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_2/Relu?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_3/BiasAddj
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape_1/Reshapeu
IdentityIdentityreshape_1/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????+

_user_specified_namex
?u
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16284
xJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:@6
(conv1d_1_biasadd_readvariableop_resource:@;
)dense_2_tensordot_readvariableop_resource:@ 5
'dense_2_biasadd_readvariableop_resource: ;
)dense_3_tensordot_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource:
identity??conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp? dense_3/Tensordot/ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDimsstrided_slice:output:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
conv1d_1/Reluw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulconv1d_1/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????@2
dropout_2/dropout/Mul}
dropout_2/dropout/ShapeShapeconv1d_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0*

seed*20
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@2
dropout_2/dropout/Mul_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free}
dense_2/Tensordot/ShapeShapedropout_2/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transposedropout_2/dropout/Mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
dense_2/BiasAddt
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
dense_2/Relu?
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes?
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free|
dense_3/Tensordot/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape?
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis?
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2?
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis?
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const?
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod?
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1?
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1?
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis?
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat?
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack?
dense_3/Tensordot/transpose	Transposedense_2/Relu:activations:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
dense_3/Tensordot/transpose?
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_3/Tensordot/Reshape?
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/Tensordot/MatMul?
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/Const_2?
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis?
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1?
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_3/Tensordot?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_3/BiasAddj
reshape_1/ShapeShapedense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_3/BiasAdd:output:0 reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
reshape_1/Reshapeu
IdentityIdentityreshape_1/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????+

_user_specified_namex
?
?
'__inference_dense_2_layer_call_fn_16383

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_158042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_15858

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_15975
x$
conv1d_1_15938:@
conv1d_1_15940:@
dense_2_15957:@ 
dense_2_15959: 
dense_3_15968: 
dense_3_15970:
identity?? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_1_15938conv1d_1_15940*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_157672"
 conv1d_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_159552#
!dropout_2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_15957dense_2_15959*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_158042!
dense_2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_159662
dropout_3/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_15968dense_3_15970*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_158402!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_158582
reshape_1/PartitionedCall}
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:N J
+
_output_shapes
:?????????+

_user_specified_namex
?t
?
!__inference__traced_restore_16668
file_prefixS
=assignvariableop_seq_seq_single_sations_cnn_1_conv1d_1_kernel:@K
=assignvariableop_1_seq_seq_single_sations_cnn_1_conv1d_1_bias:@P
>assignvariableop_2_seq_seq_single_sations_cnn_1_dense_2_kernel:@ J
<assignvariableop_3_seq_seq_single_sations_cnn_1_dense_2_bias: P
>assignvariableop_4_seq_seq_single_sations_cnn_1_dense_3_kernel: J
<assignvariableop_5_seq_seq_single_sations_cnn_1_dense_3_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: ]
Gassignvariableop_13_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_m:@S
Eassignvariableop_14_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_m:@X
Fassignvariableop_15_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_m:@ R
Dassignvariableop_16_adam_seq_seq_single_sations_cnn_1_dense_2_bias_m: X
Fassignvariableop_17_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_m: R
Dassignvariableop_18_adam_seq_seq_single_sations_cnn_1_dense_3_bias_m:]
Gassignvariableop_19_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_v:@S
Eassignvariableop_20_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_v:@X
Fassignvariableop_21_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_v:@ R
Dassignvariableop_22_adam_seq_seq_single_sations_cnn_1_dense_2_bias_v: X
Fassignvariableop_23_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_v: R
Dassignvariableop_24_adam_seq_seq_single_sations_cnn_1_dense_3_bias_v:
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%cnn/kernel/.ATTRIBUTES/VARIABLE_VALUEB#cnn/bias/.ATTRIBUTES/VARIABLE_VALUEB.hidden_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB,hidden_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAcnn/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?cnn/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEout_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAcnn/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?cnn/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEout_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp=assignvariableop_seq_seq_single_sations_cnn_1_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp=assignvariableop_1_seq_seq_single_sations_cnn_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp>assignvariableop_2_seq_seq_single_sations_cnn_1_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp<assignvariableop_3_seq_seq_single_sations_cnn_1_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp>assignvariableop_4_seq_seq_single_sations_cnn_1_dense_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp<assignvariableop_5_seq_seq_single_sations_cnn_1_dense_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpGassignvariableop_13_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpEassignvariableop_14_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpFassignvariableop_15_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpDassignvariableop_16_adam_seq_seq_single_sations_cnn_1_dense_2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpFassignvariableop_17_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpDassignvariableop_18_adam_seq_seq_single_sations_cnn_1_dense_3_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpGassignvariableop_19_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpEassignvariableop_20_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpFassignvariableop_21_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpDassignvariableop_22_adam_seq_seq_single_sations_cnn_1_dense_2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpFassignvariableop_23_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpDassignvariableop_24_adam_seq_seq_single_sations_cnn_1_dense_3_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25f
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_26?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
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
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_16439

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????@2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
? 
?
B__inference_dense_3_layer_call_and_return_conditional_losses_15840

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_15955

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_15998

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?A
?
__inference__traced_save_16583
file_prefixK
Gsavev2_seq_seq_single_sations_cnn_1_conv1d_1_kernel_read_readvariableopI
Esavev2_seq_seq_single_sations_cnn_1_conv1d_1_bias_read_readvariableopJ
Fsavev2_seq_seq_single_sations_cnn_1_dense_2_kernel_read_readvariableopH
Dsavev2_seq_seq_single_sations_cnn_1_dense_2_bias_read_readvariableopJ
Fsavev2_seq_seq_single_sations_cnn_1_dense_3_kernel_read_readvariableopH
Dsavev2_seq_seq_single_sations_cnn_1_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopR
Nsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_m_read_readvariableopP
Lsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_m_read_readvariableopQ
Msavev2_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_m_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_1_dense_2_bias_m_read_readvariableopQ
Msavev2_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_m_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_1_dense_3_bias_m_read_readvariableopR
Nsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_v_read_readvariableopP
Lsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_v_read_readvariableopQ
Msavev2_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_v_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_1_dense_2_bias_v_read_readvariableopQ
Msavev2_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_v_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_1_dense_3_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%cnn/kernel/.ATTRIBUTES/VARIABLE_VALUEB#cnn/bias/.ATTRIBUTES/VARIABLE_VALUEB.hidden_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB,hidden_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBAcnn/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB?cnn/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBEout_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBAcnn/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB?cnn/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBEout_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Gsavev2_seq_seq_single_sations_cnn_1_conv1d_1_kernel_read_readvariableopEsavev2_seq_seq_single_sations_cnn_1_conv1d_1_bias_read_readvariableopFsavev2_seq_seq_single_sations_cnn_1_dense_2_kernel_read_readvariableopDsavev2_seq_seq_single_sations_cnn_1_dense_2_bias_read_readvariableopFsavev2_seq_seq_single_sations_cnn_1_dense_3_kernel_read_readvariableopDsavev2_seq_seq_single_sations_cnn_1_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopNsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_m_read_readvariableopLsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_m_read_readvariableopMsavev2_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_m_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_1_dense_2_bias_m_read_readvariableopMsavev2_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_m_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_1_dense_3_bias_m_read_readvariableopNsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_kernel_v_read_readvariableopLsavev2_adam_seq_seq_single_sations_cnn_1_conv1d_1_bias_v_read_readvariableopMsavev2_adam_seq_seq_single_sations_cnn_1_dense_2_kernel_v_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_1_dense_2_bias_v_read_readvariableopMsavev2_adam_seq_seq_single_sations_cnn_1_dense_3_kernel_v_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_1_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@ : : :: : : : : : : :@:@:@ : : ::@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	
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
: :($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::($
"
_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16063
input_1$
conv1d_1_16046:@
conv1d_1_16048:@
dense_2_16051:@ 
dense_2_16053: 
dense_3_16056: 
dense_3_16058:
identity?? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_1_16046conv1d_1_16048*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_157672"
 conv1d_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0dense_2_16051dense_2_16053*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_158042!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_16056dense_3_16058*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_158402!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_158582
reshape_1/PartitionedCall}
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
ͥ
?
 __inference__wrapped_model_15740
input_1g
Qseq_seq_single_sations_cnn_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource:@S
Eseq_seq_single_sations_cnn_1_conv1d_1_biasadd_readvariableop_resource:@X
Fseq_seq_single_sations_cnn_1_dense_2_tensordot_readvariableop_resource:@ R
Dseq_seq_single_sations_cnn_1_dense_2_biasadd_readvariableop_resource: X
Fseq_seq_single_sations_cnn_1_dense_3_tensordot_readvariableop_resource: R
Dseq_seq_single_sations_cnn_1_dense_3_biasadd_readvariableop_resource:
identity??<seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOp?Hseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?;seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp?=seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp?;seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp?=seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp?
0seq_seq_single_sations_cnn_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           22
0seq_seq_single_sations_cnn_1/strided_slice/stack?
2seq_seq_single_sations_cnn_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            24
2seq_seq_single_sations_cnn_1/strided_slice/stack_1?
2seq_seq_single_sations_cnn_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2seq_seq_single_sations_cnn_1/strided_slice/stack_2?
*seq_seq_single_sations_cnn_1/strided_sliceStridedSliceinput_19seq_seq_single_sations_cnn_1/strided_slice/stack:output:0;seq_seq_single_sations_cnn_1/strided_slice/stack_1:output:0;seq_seq_single_sations_cnn_1/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2,
*seq_seq_single_sations_cnn_1/strided_slice?
;seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2=
;seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims/dim?
7seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims
ExpandDims3seq_seq_single_sations_cnn_1/strided_slice:output:0Dseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????29
7seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims?
Hseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpQseq_seq_single_sations_cnn_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02J
Hseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
=seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2?
=seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/dim?
9seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1
ExpandDimsPseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0Fseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2;
9seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1?
,seq_seq_single_sations_cnn_1/conv1d_1/conv1dConv2D@seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims:output:0Bseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2.
,seq_seq_single_sations_cnn_1/conv1d_1/conv1d?
4seq_seq_single_sations_cnn_1/conv1d_1/conv1d/SqueezeSqueeze5seq_seq_single_sations_cnn_1/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????26
4seq_seq_single_sations_cnn_1/conv1d_1/conv1d/Squeeze?
<seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpEseq_seq_single_sations_cnn_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02>
<seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOp?
-seq_seq_single_sations_cnn_1/conv1d_1/BiasAddBiasAdd=seq_seq_single_sations_cnn_1/conv1d_1/conv1d/Squeeze:output:0Dseq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2/
-seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd?
*seq_seq_single_sations_cnn_1/conv1d_1/ReluRelu6seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@2,
*seq_seq_single_sations_cnn_1/conv1d_1/Relu?
=seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOpReadVariableOpFseq_seq_single_sations_cnn_1_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02?
=seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp?
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/axes?
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/free?
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/ShapeShape8seq_seq_single_sations_cnn_1/conv1d_1/Relu:activations:0*
T0*
_output_shapes
:26
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/Shape?
<seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2/axis?
7seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2GatherV2=seq_seq_single_sations_cnn_1/dense_2/Tensordot/Shape:output:0<seq_seq_single_sations_cnn_1/dense_2/Tensordot/free:output:0Eseq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2?
>seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1/axis?
9seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1GatherV2=seq_seq_single_sations_cnn_1/dense_2/Tensordot/Shape:output:0<seq_seq_single_sations_cnn_1/dense_2/Tensordot/axes:output:0Gseq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1?
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const?
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/ProdProd@seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2:output:0=seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3seq_seq_single_sations_cnn_1/dense_2/Tensordot/Prod?
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_1?
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/Prod_1ProdBseq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2_1:output:0?seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/Prod_1?
:seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat/axis?
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/concatConcatV2<seq_seq_single_sations_cnn_1/dense_2/Tensordot/free:output:0<seq_seq_single_sations_cnn_1/dense_2/Tensordot/axes:output:0Cseq_seq_single_sations_cnn_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat?
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/stackPack<seq_seq_single_sations_cnn_1/dense_2/Tensordot/Prod:output:0>seq_seq_single_sations_cnn_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4seq_seq_single_sations_cnn_1/dense_2/Tensordot/stack?
8seq_seq_single_sations_cnn_1/dense_2/Tensordot/transpose	Transpose8seq_seq_single_sations_cnn_1/conv1d_1/Relu:activations:0>seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????@2:
8seq_seq_single_sations_cnn_1/dense_2/Tensordot/transpose?
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReshapeReshape<seq_seq_single_sations_cnn_1/dense_2/Tensordot/transpose:y:0=seq_seq_single_sations_cnn_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/Reshape?
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/MatMulMatMul?seq_seq_single_sations_cnn_1/dense_2/Tensordot/Reshape:output:0Eseq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 27
5seq_seq_single_sations_cnn_1/dense_2/Tensordot/MatMul?
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 28
6seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_2?
<seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1/axis?
7seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1ConcatV2@seq_seq_single_sations_cnn_1/dense_2/Tensordot/GatherV2:output:0?seq_seq_single_sations_cnn_1/dense_2/Tensordot/Const_2:output:0Eseq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1?
.seq_seq_single_sations_cnn_1/dense_2/TensordotReshape?seq_seq_single_sations_cnn_1/dense_2/Tensordot/MatMul:product:0@seq_seq_single_sations_cnn_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 20
.seq_seq_single_sations_cnn_1/dense_2/Tensordot?
;seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpDseq_seq_single_sations_cnn_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp?
,seq_seq_single_sations_cnn_1/dense_2/BiasAddBiasAdd7seq_seq_single_sations_cnn_1/dense_2/Tensordot:output:0Cseq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2.
,seq_seq_single_sations_cnn_1/dense_2/BiasAdd?
)seq_seq_single_sations_cnn_1/dense_2/ReluRelu5seq_seq_single_sations_cnn_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2+
)seq_seq_single_sations_cnn_1/dense_2/Relu?
=seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOpReadVariableOpFseq_seq_single_sations_cnn_1_dense_3_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02?
=seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp?
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/axes?
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/free?
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/ShapeShape7seq_seq_single_sations_cnn_1/dense_2/Relu:activations:0*
T0*
_output_shapes
:26
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/Shape?
<seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2/axis?
7seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2GatherV2=seq_seq_single_sations_cnn_1/dense_3/Tensordot/Shape:output:0<seq_seq_single_sations_cnn_1/dense_3/Tensordot/free:output:0Eseq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2?
>seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1/axis?
9seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1GatherV2=seq_seq_single_sations_cnn_1/dense_3/Tensordot/Shape:output:0<seq_seq_single_sations_cnn_1/dense_3/Tensordot/axes:output:0Gseq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1?
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const?
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/ProdProd@seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2:output:0=seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3seq_seq_single_sations_cnn_1/dense_3/Tensordot/Prod?
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_1?
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/Prod_1ProdBseq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2_1:output:0?seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/Prod_1?
:seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat/axis?
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/concatConcatV2<seq_seq_single_sations_cnn_1/dense_3/Tensordot/free:output:0<seq_seq_single_sations_cnn_1/dense_3/Tensordot/axes:output:0Cseq_seq_single_sations_cnn_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat?
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/stackPack<seq_seq_single_sations_cnn_1/dense_3/Tensordot/Prod:output:0>seq_seq_single_sations_cnn_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4seq_seq_single_sations_cnn_1/dense_3/Tensordot/stack?
8seq_seq_single_sations_cnn_1/dense_3/Tensordot/transpose	Transpose7seq_seq_single_sations_cnn_1/dense_2/Relu:activations:0>seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:????????? 2:
8seq_seq_single_sations_cnn_1/dense_3/Tensordot/transpose?
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReshapeReshape<seq_seq_single_sations_cnn_1/dense_3/Tensordot/transpose:y:0=seq_seq_single_sations_cnn_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????28
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/Reshape?
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/MatMulMatMul?seq_seq_single_sations_cnn_1/dense_3/Tensordot/Reshape:output:0Eseq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????27
5seq_seq_single_sations_cnn_1/dense_3/Tensordot/MatMul?
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_2?
<seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1/axis?
7seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1ConcatV2@seq_seq_single_sations_cnn_1/dense_3/Tensordot/GatherV2:output:0?seq_seq_single_sations_cnn_1/dense_3/Tensordot/Const_2:output:0Eseq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1?
.seq_seq_single_sations_cnn_1/dense_3/TensordotReshape?seq_seq_single_sations_cnn_1/dense_3/Tensordot/MatMul:product:0@seq_seq_single_sations_cnn_1/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????20
.seq_seq_single_sations_cnn_1/dense_3/Tensordot?
;seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOpReadVariableOpDseq_seq_single_sations_cnn_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp?
,seq_seq_single_sations_cnn_1/dense_3/BiasAddBiasAdd7seq_seq_single_sations_cnn_1/dense_3/Tensordot:output:0Cseq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2.
,seq_seq_single_sations_cnn_1/dense_3/BiasAdd?
,seq_seq_single_sations_cnn_1/reshape_1/ShapeShape5seq_seq_single_sations_cnn_1/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2.
,seq_seq_single_sations_cnn_1/reshape_1/Shape?
:seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack?
<seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_1?
<seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<seq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_2?
4seq_seq_single_sations_cnn_1/reshape_1/strided_sliceStridedSlice5seq_seq_single_sations_cnn_1/reshape_1/Shape:output:0Cseq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack:output:0Eseq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_1:output:0Eseq_seq_single_sations_cnn_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4seq_seq_single_sations_cnn_1/reshape_1/strided_slice?
6seq_seq_single_sations_cnn_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6seq_seq_single_sations_cnn_1/reshape_1/Reshape/shape/1?
4seq_seq_single_sations_cnn_1/reshape_1/Reshape/shapePack=seq_seq_single_sations_cnn_1/reshape_1/strided_slice:output:0?seq_seq_single_sations_cnn_1/reshape_1/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:26
4seq_seq_single_sations_cnn_1/reshape_1/Reshape/shape?
.seq_seq_single_sations_cnn_1/reshape_1/ReshapeReshape5seq_seq_single_sations_cnn_1/dense_3/BiasAdd:output:0=seq_seq_single_sations_cnn_1/reshape_1/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????20
.seq_seq_single_sations_cnn_1/reshape_1/Reshape?
IdentityIdentity7seq_seq_single_sations_cnn_1/reshape_1/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp=^seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOpI^seq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp<^seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp>^seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp<^seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp>^seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2|
<seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOp<seq_seq_single_sations_cnn_1/conv1d_1/BiasAdd/ReadVariableOp2?
Hseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpHseq_seq_single_sations_cnn_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2z
;seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp;seq_seq_single_sations_cnn_1/dense_2/BiasAdd/ReadVariableOp2~
=seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp=seq_seq_single_sations_cnn_1/dense_2/Tensordot/ReadVariableOp2z
;seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp;seq_seq_single_sations_cnn_1/dense_3/BiasAdd/ReadVariableOp2~
=seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp=seq_seq_single_sations_cnn_1/dense_3/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
?
?
'__inference_dense_3_layer_call_fn_16422

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_158402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
B__inference_dense_2_layer_call_and_return_conditional_losses_15804

inputs3
!tensordot_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@ *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:????????? 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?"
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16089
input_1$
conv1d_1_16070:@
conv1d_1_16072:@
dense_2_16076:@ 
dense_2_16078: 
dense_3_16082: 
dense_3_16084:
identity?? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_1_16070conv1d_1_16072*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_157672"
 conv1d_1/StatefulPartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_159552#
!dropout_2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_2_16076dense_2_16078*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_158042!
dense_2/StatefulPartitionedCall?
dropout_3/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_159662
dropout_3/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_3_16082dense_3_16084*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_158402!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_158582
reshape_1/PartitionedCall}
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
?	
?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16318
x
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_159752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????+

_user_specified_namex
?
`
D__inference_dropout_3_layer_call_and_return_conditional_losses_15966

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_16449

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_159552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_16334

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_1_layer_call_fn_16485

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_158582
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_3_layer_call_fn_16468

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_159662
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_dropout_3_layer_call_and_return_conditional_losses_16453

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_16480

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_15767

inputsA
+conv1d_expanddims_1_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16039
input_1
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_159752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????+
!
_user_specified_name	input_1
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_16434

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0*

seed*2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_15861
x$
conv1d_1_15768:@
conv1d_1_15770:@
dense_2_15805:@ 
dense_2_15807: 
dense_3_15841: 
dense_3_15843:
identity?? conv1d_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSlicexstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????*

begin_mask*
end_mask2
strided_slice?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_1_15768conv1d_1_15770*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_157672"
 conv1d_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0dense_2_15805dense_2_15807*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_158042!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_15841dense_3_15843*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_158402!
dense_3/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_158582
reshape_1/PartitionedCall}
IdentityIdentity"reshape_1/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^conv1d_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????+: : : : : : 2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:N J
+
_output_shapes
:?????????+

_user_specified_namex
?
E
)__inference_dropout_2_layer_call_fn_16444

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_160142
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????+<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?}
?
cnn
batch_norm_layer
hidden_dense
	out_dense
conv_dropout
dense_dropout
reshape_output
	optimizer
		variables

regularization_losses
trainable_variables
	keras_api

signatures
*f&call_and_return_all_conditional_losses
g_default_save_signature
h__call__"
_tf_keras_model
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*k&call_and_return_all_conditional_losses
l__call__"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
 	keras_api
*m&call_and_return_all_conditional_losses
n__call__"
_tf_keras_layer
?
!	variables
"regularization_losses
#trainable_variables
$	keras_api
*o&call_and_return_all_conditional_losses
p__call__"
_tf_keras_layer
?
%	variables
&regularization_losses
'trainable_variables
(	keras_api
*q&call_and_return_all_conditional_losses
r__call__"
_tf_keras_layer
?
)	variables
*regularization_losses
+trainable_variables
,	keras_api
*s&call_and_return_all_conditional_losses
t__call__"
_tf_keras_layer
?
-iter

.beta_1

/beta_2
	0decay
1learning_ratemZm[m\m]m^m_v`vavbvcvdve"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
2metrics
		variables
3non_trainable_variables

regularization_losses

4layers
5layer_regularization_losses
trainable_variables
6layer_metrics
h__call__
g_default_save_signature
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
,
userving_default"
signature_map
B:@@2,seq_seq_single_sations_cnn_1/conv1d_1/kernel
8:6@2*seq_seq_single_sations_cnn_1/conv1d_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
7metrics
	variables
8non_trainable_variables
regularization_losses

9layers
:layer_regularization_losses
trainable_variables
;layer_metrics
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
=:;@ 2+seq_seq_single_sations_cnn_1/dense_2/kernel
7:5 2)seq_seq_single_sations_cnn_1/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
<metrics
	variables
=non_trainable_variables
regularization_losses

>layers
?layer_regularization_losses
trainable_variables
@layer_metrics
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
=:; 2+seq_seq_single_sations_cnn_1/dense_3/kernel
7:52)seq_seq_single_sations_cnn_1/dense_3/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Ametrics
	variables
Bnon_trainable_variables
regularization_losses

Clayers
Dlayer_regularization_losses
trainable_variables
Elayer_metrics
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fmetrics
!	variables
Gnon_trainable_variables
"regularization_losses

Hlayers
Ilayer_regularization_losses
#trainable_variables
Jlayer_metrics
p__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Kmetrics
%	variables
Lnon_trainable_variables
&regularization_losses

Mlayers
Nlayer_regularization_losses
'trainable_variables
Olayer_metrics
r__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pmetrics
)	variables
Qnon_trainable_variables
*regularization_losses

Rlayers
Slayer_regularization_losses
+trainable_variables
Tlayer_metrics
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
'
U0"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
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
N
	Vtotal
	Wcount
X	variables
Y	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
V0
W1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
G:E@23Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/m
=:;@21Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/m
B:@@ 22Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/m
<:: 20Adam/seq_seq_single_sations_cnn_1/dense_2/bias/m
B:@ 22Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/m
<::20Adam/seq_seq_single_sations_cnn_1/dense_3/bias/m
G:E@23Adam/seq_seq_single_sations_cnn_1/conv1d_1/kernel/v
=:;@21Adam/seq_seq_single_sations_cnn_1/conv1d_1/bias/v
B:@@ 22Adam/seq_seq_single_sations_cnn_1/dense_2/kernel/v
<:: 20Adam/seq_seq_single_sations_cnn_1/dense_2/bias/v
B:@ 22Adam/seq_seq_single_sations_cnn_1/dense_3/kernel/v
<::20Adam/seq_seq_single_sations_cnn_1/dense_3/bias/v
?2?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16195
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16284
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16063
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16089?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_15740input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_15876
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16301
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16318
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16039?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_16334?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv1d_1_layer_call_fn_16343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_16374?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_16383?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_16413?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_3_layer_call_fn_16422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_16434
D__inference_dropout_2_layer_call_and_return_conditional_losses_16439?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_16444
)__inference_dropout_2_layer_call_fn_16449?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dropout_3_layer_call_and_return_conditional_losses_16453
D__inference_dropout_3_layer_call_and_return_conditional_losses_16458?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_3_layer_call_fn_16463
)__inference_dropout_3_layer_call_fn_16468?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_reshape_1_layer_call_and_return_conditional_losses_16480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_1_layer_call_fn_16485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_16114input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_15740s4?1
*?'
%?"
input_1?????????+
? "3?0
.
output_1"?
output_1??????????
C__inference_conv1d_1_layer_call_and_return_conditional_losses_16334d3?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????@
? ?
(__inference_conv1d_1_layer_call_fn_16343W3?0
)?&
$?!
inputs?????????
? "??????????@?
B__inference_dense_2_layer_call_and_return_conditional_losses_16374d3?0
)?&
$?!
inputs?????????@
? ")?&
?
0????????? 
? ?
'__inference_dense_2_layer_call_fn_16383W3?0
)?&
$?!
inputs?????????@
? "?????????? ?
B__inference_dense_3_layer_call_and_return_conditional_losses_16413d3?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????
? ?
'__inference_dense_3_layer_call_fn_16422W3?0
)?&
$?!
inputs????????? 
? "???????????
D__inference_dropout_2_layer_call_and_return_conditional_losses_16434d7?4
-?*
$?!
inputs?????????@
p
? ")?&
?
0?????????@
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_16439d7?4
-?*
$?!
inputs?????????@
p 
? ")?&
?
0?????????@
? ?
)__inference_dropout_2_layer_call_fn_16444W7?4
-?*
$?!
inputs?????????@
p 
? "??????????@?
)__inference_dropout_2_layer_call_fn_16449W7?4
-?*
$?!
inputs?????????@
p
? "??????????@?
D__inference_dropout_3_layer_call_and_return_conditional_losses_16453d7?4
-?*
$?!
inputs????????? 
p
? ")?&
?
0????????? 
? ?
D__inference_dropout_3_layer_call_and_return_conditional_losses_16458d7?4
-?*
$?!
inputs????????? 
p 
? ")?&
?
0????????? 
? ?
)__inference_dropout_3_layer_call_fn_16463W7?4
-?*
$?!
inputs????????? 
p 
? "?????????? ?
)__inference_dropout_3_layer_call_fn_16468W7?4
-?*
$?!
inputs????????? 
p
? "?????????? ?
D__inference_reshape_1_layer_call_and_return_conditional_losses_16480\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? |
)__inference_reshape_1_layer_call_fn_16485O3?0
)?&
$?!
inputs?????????
? "???????????
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16063i8?5
.?+
%?"
input_1?????????+
p 
? "%?"
?
0?????????
? ?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16089i8?5
.?+
%?"
input_1?????????+
p
? "%?"
?
0?????????
? ?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16195c2?/
(?%
?
x?????????+
p 
? "%?"
?
0?????????
? ?
W__inference_seq_seq_single_sations_cnn_1_layer_call_and_return_conditional_losses_16284c2?/
(?%
?
x?????????+
p
? "%?"
?
0?????????
? ?
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_15876\8?5
.?+
%?"
input_1?????????+
p 
? "???????????
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16039\8?5
.?+
%?"
input_1?????????+
p
? "???????????
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16301V2?/
(?%
?
x?????????+
p 
? "???????????
<__inference_seq_seq_single_sations_cnn_1_layer_call_fn_16318V2?/
(?%
?
x?????????+
p
? "???????????
#__inference_signature_wrapper_16114~??<
? 
5?2
0
input_1%?"
input_1?????????+"3?0
.
output_1"?
output_1?????????