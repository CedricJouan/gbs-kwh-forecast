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
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
(seq_seq_single_sations_cnn/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(seq_seq_single_sations_cnn/conv1d/kernel
?
<seq_seq_single_sations_cnn/conv1d/kernel/Read/ReadVariableOpReadVariableOp(seq_seq_single_sations_cnn/conv1d/kernel*"
_output_shapes
: *
dtype0
?
&seq_seq_single_sations_cnn/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&seq_seq_single_sations_cnn/conv1d/bias
?
:seq_seq_single_sations_cnn/conv1d/bias/Read/ReadVariableOpReadVariableOp&seq_seq_single_sations_cnn/conv1d/bias*
_output_shapes
: *
dtype0
?
'seq_seq_single_sations_cnn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *8
shared_name)'seq_seq_single_sations_cnn/dense/kernel
?
;seq_seq_single_sations_cnn/dense/kernel/Read/ReadVariableOpReadVariableOp'seq_seq_single_sations_cnn/dense/kernel*
_output_shapes

:  *
dtype0
?
%seq_seq_single_sations_cnn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%seq_seq_single_sations_cnn/dense/bias
?
9seq_seq_single_sations_cnn/dense/bias/Read/ReadVariableOpReadVariableOp%seq_seq_single_sations_cnn/dense/bias*
_output_shapes
: *
dtype0
?
)seq_seq_single_sations_cnn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)seq_seq_single_sations_cnn/dense_1/kernel
?
=seq_seq_single_sations_cnn/dense_1/kernel/Read/ReadVariableOpReadVariableOp)seq_seq_single_sations_cnn/dense_1/kernel*
_output_shapes

: *
dtype0
?
'seq_seq_single_sations_cnn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'seq_seq_single_sations_cnn/dense_1/bias
?
;seq_seq_single_sations_cnn/dense_1/bias/Read/ReadVariableOpReadVariableOp'seq_seq_single_sations_cnn/dense_1/bias*
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
/Adam/seq_seq_single_sations_cnn/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/seq_seq_single_sations_cnn/conv1d/kernel/m
?
CAdam/seq_seq_single_sations_cnn/conv1d/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/seq_seq_single_sations_cnn/conv1d/kernel/m*"
_output_shapes
: *
dtype0
?
-Adam/seq_seq_single_sations_cnn/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/seq_seq_single_sations_cnn/conv1d/bias/m
?
AAdam/seq_seq_single_sations_cnn/conv1d/bias/m/Read/ReadVariableOpReadVariableOp-Adam/seq_seq_single_sations_cnn/conv1d/bias/m*
_output_shapes
: *
dtype0
?
.Adam/seq_seq_single_sations_cnn/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/seq_seq_single_sations_cnn/dense/kernel/m
?
BAdam/seq_seq_single_sations_cnn/dense/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/seq_seq_single_sations_cnn/dense/kernel/m*
_output_shapes

:  *
dtype0
?
,Adam/seq_seq_single_sations_cnn/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/seq_seq_single_sations_cnn/dense/bias/m
?
@Adam/seq_seq_single_sations_cnn/dense/bias/m/Read/ReadVariableOpReadVariableOp,Adam/seq_seq_single_sations_cnn/dense/bias/m*
_output_shapes
: *
dtype0
?
0Adam/seq_seq_single_sations_cnn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20Adam/seq_seq_single_sations_cnn/dense_1/kernel/m
?
DAdam/seq_seq_single_sations_cnn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn/dense_1/kernel/m*
_output_shapes

: *
dtype0
?
.Adam/seq_seq_single_sations_cnn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/seq_seq_single_sations_cnn/dense_1/bias/m
?
BAdam/seq_seq_single_sations_cnn/dense_1/bias/m/Read/ReadVariableOpReadVariableOp.Adam/seq_seq_single_sations_cnn/dense_1/bias/m*
_output_shapes
:*
dtype0
?
/Adam/seq_seq_single_sations_cnn/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/seq_seq_single_sations_cnn/conv1d/kernel/v
?
CAdam/seq_seq_single_sations_cnn/conv1d/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/seq_seq_single_sations_cnn/conv1d/kernel/v*"
_output_shapes
: *
dtype0
?
-Adam/seq_seq_single_sations_cnn/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/seq_seq_single_sations_cnn/conv1d/bias/v
?
AAdam/seq_seq_single_sations_cnn/conv1d/bias/v/Read/ReadVariableOpReadVariableOp-Adam/seq_seq_single_sations_cnn/conv1d/bias/v*
_output_shapes
: *
dtype0
?
.Adam/seq_seq_single_sations_cnn/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *?
shared_name0.Adam/seq_seq_single_sations_cnn/dense/kernel/v
?
BAdam/seq_seq_single_sations_cnn/dense/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/seq_seq_single_sations_cnn/dense/kernel/v*
_output_shapes

:  *
dtype0
?
,Adam/seq_seq_single_sations_cnn/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,Adam/seq_seq_single_sations_cnn/dense/bias/v
?
@Adam/seq_seq_single_sations_cnn/dense/bias/v/Read/ReadVariableOpReadVariableOp,Adam/seq_seq_single_sations_cnn/dense/bias/v*
_output_shapes
: *
dtype0
?
0Adam/seq_seq_single_sations_cnn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20Adam/seq_seq_single_sations_cnn/dense_1/kernel/v
?
DAdam/seq_seq_single_sations_cnn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/seq_seq_single_sations_cnn/dense_1/kernel/v*
_output_shapes

: *
dtype0
?
.Adam/seq_seq_single_sations_cnn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/seq_seq_single_sations_cnn/dense_1/bias/v
?
BAdam/seq_seq_single_sations_cnn/dense_1/bias/v/Read/ReadVariableOpReadVariableOp.Adam/seq_seq_single_sations_cnn/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?+
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
ca
VARIABLE_VALUE(seq_seq_single_sations_cnn/conv1d/kernel%cnn/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE&seq_seq_single_sations_cnn/conv1d/bias#cnn/bias/.ATTRIBUTES/VARIABLE_VALUE
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
ki
VARIABLE_VALUE'seq_seq_single_sations_cnn/dense/kernel.hidden_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE%seq_seq_single_sations_cnn/dense/bias,hidden_dense/bias/.ATTRIBUTES/VARIABLE_VALUE
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
jh
VARIABLE_VALUE)seq_seq_single_sations_cnn/dense_1/kernel+out_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE'seq_seq_single_sations_cnn/dense_1/bias)out_dense/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUE/Adam/seq_seq_single_sations_cnn/conv1d/kernel/mAcnn/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/seq_seq_single_sations_cnn/conv1d/bias/m?cnn/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/seq_seq_single_sations_cnn/dense/kernel/mJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/seq_seq_single_sations_cnn/dense/bias/mHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn/dense_1/kernel/mGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/seq_seq_single_sations_cnn/dense_1/bias/mEout_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/seq_seq_single_sations_cnn/conv1d/kernel/vAcnn/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/seq_seq_single_sations_cnn/conv1d/bias/v?cnn/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/seq_seq_single_sations_cnn/dense/kernel/vJhidden_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/seq_seq_single_sations_cnn/dense/bias/vHhidden_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adam/seq_seq_single_sations_cnn/dense_1/kernel/vGout_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/seq_seq_single_sations_cnn/dense_1/bias/vEout_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????g*
dtype0* 
shape:?????????g
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1(seq_seq_single_sations_cnn/conv1d/kernel&seq_seq_single_sations_cnn/conv1d/bias'seq_seq_single_sations_cnn/dense/kernel%seq_seq_single_sations_cnn/dense/bias)seq_seq_single_sations_cnn/dense_1/kernel'seq_seq_single_sations_cnn/dense_1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_7708
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename<seq_seq_single_sations_cnn/conv1d/kernel/Read/ReadVariableOp:seq_seq_single_sations_cnn/conv1d/bias/Read/ReadVariableOp;seq_seq_single_sations_cnn/dense/kernel/Read/ReadVariableOp9seq_seq_single_sations_cnn/dense/bias/Read/ReadVariableOp=seq_seq_single_sations_cnn/dense_1/kernel/Read/ReadVariableOp;seq_seq_single_sations_cnn/dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpCAdam/seq_seq_single_sations_cnn/conv1d/kernel/m/Read/ReadVariableOpAAdam/seq_seq_single_sations_cnn/conv1d/bias/m/Read/ReadVariableOpBAdam/seq_seq_single_sations_cnn/dense/kernel/m/Read/ReadVariableOp@Adam/seq_seq_single_sations_cnn/dense/bias/m/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn/dense_1/kernel/m/Read/ReadVariableOpBAdam/seq_seq_single_sations_cnn/dense_1/bias/m/Read/ReadVariableOpCAdam/seq_seq_single_sations_cnn/conv1d/kernel/v/Read/ReadVariableOpAAdam/seq_seq_single_sations_cnn/conv1d/bias/v/Read/ReadVariableOpBAdam/seq_seq_single_sations_cnn/dense/kernel/v/Read/ReadVariableOp@Adam/seq_seq_single_sations_cnn/dense/bias/v/Read/ReadVariableOpDAdam/seq_seq_single_sations_cnn/dense_1/kernel/v/Read/ReadVariableOpBAdam/seq_seq_single_sations_cnn/dense_1/bias/v/Read/ReadVariableOpConst*&
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
GPU 2J 8? *&
f!R
__inference__traced_save_8161
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(seq_seq_single_sations_cnn/conv1d/kernel&seq_seq_single_sations_cnn/conv1d/bias'seq_seq_single_sations_cnn/dense/kernel%seq_seq_single_sations_cnn/dense/bias)seq_seq_single_sations_cnn/dense_1/kernel'seq_seq_single_sations_cnn/dense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount/Adam/seq_seq_single_sations_cnn/conv1d/kernel/m-Adam/seq_seq_single_sations_cnn/conv1d/bias/m.Adam/seq_seq_single_sations_cnn/dense/kernel/m,Adam/seq_seq_single_sations_cnn/dense/bias/m0Adam/seq_seq_single_sations_cnn/dense_1/kernel/m.Adam/seq_seq_single_sations_cnn/dense_1/bias/m/Adam/seq_seq_single_sations_cnn/conv1d/kernel/v-Adam/seq_seq_single_sations_cnn/conv1d/bias/v.Adam/seq_seq_single_sations_cnn/dense/kernel/v,Adam/seq_seq_single_sations_cnn/dense/bias/v0Adam/seq_seq_single_sations_cnn/dense_1/kernel/v.Adam/seq_seq_single_sations_cnn/dense_1/bias/v*%
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_8246??
?
B
&__inference_reshape_layer_call_fn_8063

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_74602
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
]
A__inference_reshape_layer_call_and_return_conditional_losses_8058

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
value	B :P2
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
:?????????P2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7887
x
unknown: 
	unknown_0: 
	unknown_1:  
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
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_74632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????g

_user_specified_namex
?
D
(__inference_dropout_1_layer_call_fn_8046

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
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75602
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?h
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7789
xH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsstrided_slice:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????V2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????P *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????P *
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
conv1d/Relu?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freew
dense/Tensordot/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposeconv1d/Relu:activations:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2
dense/BiasAddn

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freez
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :P2
reshape/Reshape/shape/1?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P2
reshape/Reshapes
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????g

_user_specified_namex
? 
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7569
x!
conv1d_7540: 
conv1d_7542: 

dense_7551:  

dense_7553: 
dense_1_7562: 
dense_1_7564:
identity??conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_7540conv1d_7542*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_73692 
conv1d/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75492
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_7551
dense_7553*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_74062
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75602
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_7562dense_1_7564*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_74422!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_74602
reshape/PartitionedCall{
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:N J
+
_output_shapes
:?????????g

_user_specified_namex
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8036

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_8022

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
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_76082
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?	
?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7633
input_1
unknown: 
	unknown_0: 
	unknown_1:  
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
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_75692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
? 
?
A__inference_dense_1_layer_call_and_return_conditional_losses_7999

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
:?????????P 2
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
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????P2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?!
?
?__inference_dense_layer_call_and_return_conditional_losses_7960

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
:?????????P 2
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
:?????????P 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
]
A__inference_dropout_layer_call_and_return_conditional_losses_8012

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
]
A__inference_reshape_layer_call_and_return_conditional_losses_7460

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
value	B :P2
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
:?????????P2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????P2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_7369

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:?????????V2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????P *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????P *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????V: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????V
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_7708
input_1
unknown: 
	unknown_0: 
	unknown_1:  
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
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_73422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
?
B
&__inference_dropout_layer_call_fn_8027

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
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75492
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
D
(__inference_dropout_1_layer_call_fn_8041

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
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75922
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
]
A__inference_dropout_layer_call_and_return_conditional_losses_7549

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
?
@__inference_conv1d_layer_call_and_return_conditional_losses_7920

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
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
:?????????V2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
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
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????P *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????P *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????V: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????V
 
_user_specified_nameinputs
?
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7463
x!
conv1d_7370: 
conv1d_7372: 

dense_7407:  

dense_7409: 
dense_1_7443: 
dense_1_7445:
identity??conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_7370conv1d_7372*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_73692 
conv1d/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0
dense_7407
dense_7409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_74062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7443dense_1_7445*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_74422!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_74602
reshape/PartitionedCall{
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:N J
+
_output_shapes
:?????????g

_user_specified_namex
? 
?
A__inference_dense_1_layer_call_and_return_conditional_losses_7442

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
:?????????P 2
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
:?????????P2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????P2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7592

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?s
?
 __inference__traced_restore_8246
file_prefixO
9assignvariableop_seq_seq_single_sations_cnn_conv1d_kernel: G
9assignvariableop_1_seq_seq_single_sations_cnn_conv1d_bias: L
:assignvariableop_2_seq_seq_single_sations_cnn_dense_kernel:  F
8assignvariableop_3_seq_seq_single_sations_cnn_dense_bias: N
<assignvariableop_4_seq_seq_single_sations_cnn_dense_1_kernel: H
:assignvariableop_5_seq_seq_single_sations_cnn_dense_1_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: Y
Cassignvariableop_13_adam_seq_seq_single_sations_cnn_conv1d_kernel_m: O
Aassignvariableop_14_adam_seq_seq_single_sations_cnn_conv1d_bias_m: T
Bassignvariableop_15_adam_seq_seq_single_sations_cnn_dense_kernel_m:  N
@assignvariableop_16_adam_seq_seq_single_sations_cnn_dense_bias_m: V
Dassignvariableop_17_adam_seq_seq_single_sations_cnn_dense_1_kernel_m: P
Bassignvariableop_18_adam_seq_seq_single_sations_cnn_dense_1_bias_m:Y
Cassignvariableop_19_adam_seq_seq_single_sations_cnn_conv1d_kernel_v: O
Aassignvariableop_20_adam_seq_seq_single_sations_cnn_conv1d_bias_v: T
Bassignvariableop_21_adam_seq_seq_single_sations_cnn_dense_kernel_v:  N
@assignvariableop_22_adam_seq_seq_single_sations_cnn_dense_bias_v: V
Dassignvariableop_23_adam_seq_seq_single_sations_cnn_dense_1_kernel_v: P
Bassignvariableop_24_adam_seq_seq_single_sations_cnn_dense_1_bias_v:
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
AssignVariableOpAssignVariableOp9assignvariableop_seq_seq_single_sations_cnn_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp9assignvariableop_1_seq_seq_single_sations_cnn_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp:assignvariableop_2_seq_seq_single_sations_cnn_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp8assignvariableop_3_seq_seq_single_sations_cnn_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp<assignvariableop_4_seq_seq_single_sations_cnn_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp:assignvariableop_5_seq_seq_single_sations_cnn_dense_1_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOpCassignvariableop_13_adam_seq_seq_single_sations_cnn_conv1d_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpAassignvariableop_14_adam_seq_seq_single_sations_cnn_conv1d_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpBassignvariableop_15_adam_seq_seq_single_sations_cnn_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp@assignvariableop_16_adam_seq_seq_single_sations_cnn_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpDassignvariableop_17_adam_seq_seq_single_sations_cnn_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpBassignvariableop_18_adam_seq_seq_single_sations_cnn_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpCassignvariableop_19_adam_seq_seq_single_sations_cnn_conv1d_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpAassignvariableop_20_adam_seq_seq_single_sations_cnn_conv1d_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpBassignvariableop_21_adam_seq_seq_single_sations_cnn_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_seq_seq_single_sations_cnn_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpDassignvariableop_23_adam_seq_seq_single_sations_cnn_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpBassignvariableop_24_adam_seq_seq_single_sations_cnn_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
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
?
_
C__inference_dropout_1_layer_call_and_return_conditional_losses_8031

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_7342
input_1c
Mseq_seq_single_sations_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource: O
Aseq_seq_single_sations_cnn_conv1d_biasadd_readvariableop_resource: T
Bseq_seq_single_sations_cnn_dense_tensordot_readvariableop_resource:  N
@seq_seq_single_sations_cnn_dense_biasadd_readvariableop_resource: V
Dseq_seq_single_sations_cnn_dense_1_tensordot_readvariableop_resource: P
Bseq_seq_single_sations_cnn_dense_1_biasadd_readvariableop_resource:
identity??8seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOp?Dseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp?7seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp?9seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp?9seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp?;seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp?
.seq_seq_single_sations_cnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           20
.seq_seq_single_sations_cnn/strided_slice/stack?
0seq_seq_single_sations_cnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"            22
0seq_seq_single_sations_cnn/strided_slice/stack_1?
0seq_seq_single_sations_cnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         22
0seq_seq_single_sations_cnn/strided_slice/stack_2?
(seq_seq_single_sations_cnn/strided_sliceStridedSliceinput_17seq_seq_single_sations_cnn/strided_slice/stack:output:09seq_seq_single_sations_cnn/strided_slice/stack_1:output:09seq_seq_single_sations_cnn/strided_slice/stack_2:output:0*
Index0*
T0*+
_output_shapes
:?????????V*

begin_mask*
end_mask2*
(seq_seq_single_sations_cnn/strided_slice?
7seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????29
7seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims/dim?
3seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims
ExpandDims1seq_seq_single_sations_cnn/strided_slice:output:0@seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????V25
3seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims?
Dseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpMseq_seq_single_sations_cnn_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02F
Dseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
9seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2;
9seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/dim?
5seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1
ExpandDimsLseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0Bseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 27
5seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1?
(seq_seq_single_sations_cnn/conv1d/conv1dConv2D<seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims:output:0>seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????P *
paddingVALID*
strides
2*
(seq_seq_single_sations_cnn/conv1d/conv1d?
0seq_seq_single_sations_cnn/conv1d/conv1d/SqueezeSqueeze1seq_seq_single_sations_cnn/conv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????P *
squeeze_dims

?????????22
0seq_seq_single_sations_cnn/conv1d/conv1d/Squeeze?
8seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOpReadVariableOpAseq_seq_single_sations_cnn_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02:
8seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOp?
)seq_seq_single_sations_cnn/conv1d/BiasAddBiasAdd9seq_seq_single_sations_cnn/conv1d/conv1d/Squeeze:output:0@seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2+
)seq_seq_single_sations_cnn/conv1d/BiasAdd?
&seq_seq_single_sations_cnn/conv1d/ReluRelu2seq_seq_single_sations_cnn/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2(
&seq_seq_single_sations_cnn/conv1d/Relu?
9seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOpReadVariableOpBseq_seq_single_sations_cnn_dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02;
9seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp?
/seq_seq_single_sations_cnn/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:21
/seq_seq_single_sations_cnn/dense/Tensordot/axes?
/seq_seq_single_sations_cnn/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       21
/seq_seq_single_sations_cnn/dense/Tensordot/free?
0seq_seq_single_sations_cnn/dense/Tensordot/ShapeShape4seq_seq_single_sations_cnn/conv1d/Relu:activations:0*
T0*
_output_shapes
:22
0seq_seq_single_sations_cnn/dense/Tensordot/Shape?
8seq_seq_single_sations_cnn/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8seq_seq_single_sations_cnn/dense/Tensordot/GatherV2/axis?
3seq_seq_single_sations_cnn/dense/Tensordot/GatherV2GatherV29seq_seq_single_sations_cnn/dense/Tensordot/Shape:output:08seq_seq_single_sations_cnn/dense/Tensordot/free:output:0Aseq_seq_single_sations_cnn/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:25
3seq_seq_single_sations_cnn/dense/Tensordot/GatherV2?
:seq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:seq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1/axis?
5seq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1GatherV29seq_seq_single_sations_cnn/dense/Tensordot/Shape:output:08seq_seq_single_sations_cnn/dense/Tensordot/axes:output:0Cseq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:27
5seq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1?
0seq_seq_single_sations_cnn/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 22
0seq_seq_single_sations_cnn/dense/Tensordot/Const?
/seq_seq_single_sations_cnn/dense/Tensordot/ProdProd<seq_seq_single_sations_cnn/dense/Tensordot/GatherV2:output:09seq_seq_single_sations_cnn/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 21
/seq_seq_single_sations_cnn/dense/Tensordot/Prod?
2seq_seq_single_sations_cnn/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 24
2seq_seq_single_sations_cnn/dense/Tensordot/Const_1?
1seq_seq_single_sations_cnn/dense/Tensordot/Prod_1Prod>seq_seq_single_sations_cnn/dense/Tensordot/GatherV2_1:output:0;seq_seq_single_sations_cnn/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 23
1seq_seq_single_sations_cnn/dense/Tensordot/Prod_1?
6seq_seq_single_sations_cnn/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 28
6seq_seq_single_sations_cnn/dense/Tensordot/concat/axis?
1seq_seq_single_sations_cnn/dense/Tensordot/concatConcatV28seq_seq_single_sations_cnn/dense/Tensordot/free:output:08seq_seq_single_sations_cnn/dense/Tensordot/axes:output:0?seq_seq_single_sations_cnn/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:23
1seq_seq_single_sations_cnn/dense/Tensordot/concat?
0seq_seq_single_sations_cnn/dense/Tensordot/stackPack8seq_seq_single_sations_cnn/dense/Tensordot/Prod:output:0:seq_seq_single_sations_cnn/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:22
0seq_seq_single_sations_cnn/dense/Tensordot/stack?
4seq_seq_single_sations_cnn/dense/Tensordot/transpose	Transpose4seq_seq_single_sations_cnn/conv1d/Relu:activations:0:seq_seq_single_sations_cnn/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 26
4seq_seq_single_sations_cnn/dense/Tensordot/transpose?
2seq_seq_single_sations_cnn/dense/Tensordot/ReshapeReshape8seq_seq_single_sations_cnn/dense/Tensordot/transpose:y:09seq_seq_single_sations_cnn/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????24
2seq_seq_single_sations_cnn/dense/Tensordot/Reshape?
1seq_seq_single_sations_cnn/dense/Tensordot/MatMulMatMul;seq_seq_single_sations_cnn/dense/Tensordot/Reshape:output:0Aseq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 23
1seq_seq_single_sations_cnn/dense/Tensordot/MatMul?
2seq_seq_single_sations_cnn/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 24
2seq_seq_single_sations_cnn/dense/Tensordot/Const_2?
8seq_seq_single_sations_cnn/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8seq_seq_single_sations_cnn/dense/Tensordot/concat_1/axis?
3seq_seq_single_sations_cnn/dense/Tensordot/concat_1ConcatV2<seq_seq_single_sations_cnn/dense/Tensordot/GatherV2:output:0;seq_seq_single_sations_cnn/dense/Tensordot/Const_2:output:0Aseq_seq_single_sations_cnn/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:25
3seq_seq_single_sations_cnn/dense/Tensordot/concat_1?
*seq_seq_single_sations_cnn/dense/TensordotReshape;seq_seq_single_sations_cnn/dense/Tensordot/MatMul:product:0<seq_seq_single_sations_cnn/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P 2,
*seq_seq_single_sations_cnn/dense/Tensordot?
7seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOpReadVariableOp@seq_seq_single_sations_cnn_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp?
(seq_seq_single_sations_cnn/dense/BiasAddBiasAdd3seq_seq_single_sations_cnn/dense/Tensordot:output:0?seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2*
(seq_seq_single_sations_cnn/dense/BiasAdd?
%seq_seq_single_sations_cnn/dense/ReluRelu1seq_seq_single_sations_cnn/dense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2'
%seq_seq_single_sations_cnn/dense/Relu?
;seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOpReadVariableOpDseq_seq_single_sations_cnn_dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02=
;seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp?
1seq_seq_single_sations_cnn/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:23
1seq_seq_single_sations_cnn/dense_1/Tensordot/axes?
1seq_seq_single_sations_cnn/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       23
1seq_seq_single_sations_cnn/dense_1/Tensordot/free?
2seq_seq_single_sations_cnn/dense_1/Tensordot/ShapeShape3seq_seq_single_sations_cnn/dense/Relu:activations:0*
T0*
_output_shapes
:24
2seq_seq_single_sations_cnn/dense_1/Tensordot/Shape?
:seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2/axis?
5seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2GatherV2;seq_seq_single_sations_cnn/dense_1/Tensordot/Shape:output:0:seq_seq_single_sations_cnn/dense_1/Tensordot/free:output:0Cseq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:27
5seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2?
<seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1/axis?
7seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1GatherV2;seq_seq_single_sations_cnn/dense_1/Tensordot/Shape:output:0:seq_seq_single_sations_cnn/dense_1/Tensordot/axes:output:0Eseq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1?
2seq_seq_single_sations_cnn/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 24
2seq_seq_single_sations_cnn/dense_1/Tensordot/Const?
1seq_seq_single_sations_cnn/dense_1/Tensordot/ProdProd>seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2:output:0;seq_seq_single_sations_cnn/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 23
1seq_seq_single_sations_cnn/dense_1/Tensordot/Prod?
4seq_seq_single_sations_cnn/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 26
4seq_seq_single_sations_cnn/dense_1/Tensordot/Const_1?
3seq_seq_single_sations_cnn/dense_1/Tensordot/Prod_1Prod@seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2_1:output:0=seq_seq_single_sations_cnn/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 25
3seq_seq_single_sations_cnn/dense_1/Tensordot/Prod_1?
8seq_seq_single_sations_cnn/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8seq_seq_single_sations_cnn/dense_1/Tensordot/concat/axis?
3seq_seq_single_sations_cnn/dense_1/Tensordot/concatConcatV2:seq_seq_single_sations_cnn/dense_1/Tensordot/free:output:0:seq_seq_single_sations_cnn/dense_1/Tensordot/axes:output:0Aseq_seq_single_sations_cnn/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:25
3seq_seq_single_sations_cnn/dense_1/Tensordot/concat?
2seq_seq_single_sations_cnn/dense_1/Tensordot/stackPack:seq_seq_single_sations_cnn/dense_1/Tensordot/Prod:output:0<seq_seq_single_sations_cnn/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:24
2seq_seq_single_sations_cnn/dense_1/Tensordot/stack?
6seq_seq_single_sations_cnn/dense_1/Tensordot/transpose	Transpose3seq_seq_single_sations_cnn/dense/Relu:activations:0<seq_seq_single_sations_cnn/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 28
6seq_seq_single_sations_cnn/dense_1/Tensordot/transpose?
4seq_seq_single_sations_cnn/dense_1/Tensordot/ReshapeReshape:seq_seq_single_sations_cnn/dense_1/Tensordot/transpose:y:0;seq_seq_single_sations_cnn/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????26
4seq_seq_single_sations_cnn/dense_1/Tensordot/Reshape?
3seq_seq_single_sations_cnn/dense_1/Tensordot/MatMulMatMul=seq_seq_single_sations_cnn/dense_1/Tensordot/Reshape:output:0Cseq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????25
3seq_seq_single_sations_cnn/dense_1/Tensordot/MatMul?
4seq_seq_single_sations_cnn/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:26
4seq_seq_single_sations_cnn/dense_1/Tensordot/Const_2?
:seq_seq_single_sations_cnn/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:seq_seq_single_sations_cnn/dense_1/Tensordot/concat_1/axis?
5seq_seq_single_sations_cnn/dense_1/Tensordot/concat_1ConcatV2>seq_seq_single_sations_cnn/dense_1/Tensordot/GatherV2:output:0=seq_seq_single_sations_cnn/dense_1/Tensordot/Const_2:output:0Cseq_seq_single_sations_cnn/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:27
5seq_seq_single_sations_cnn/dense_1/Tensordot/concat_1?
,seq_seq_single_sations_cnn/dense_1/TensordotReshape=seq_seq_single_sations_cnn/dense_1/Tensordot/MatMul:product:0>seq_seq_single_sations_cnn/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2.
,seq_seq_single_sations_cnn/dense_1/Tensordot?
9seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOpReadVariableOpBseq_seq_single_sations_cnn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp?
*seq_seq_single_sations_cnn/dense_1/BiasAddBiasAdd5seq_seq_single_sations_cnn/dense_1/Tensordot:output:0Aseq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2,
*seq_seq_single_sations_cnn/dense_1/BiasAdd?
(seq_seq_single_sations_cnn/reshape/ShapeShape3seq_seq_single_sations_cnn/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2*
(seq_seq_single_sations_cnn/reshape/Shape?
6seq_seq_single_sations_cnn/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6seq_seq_single_sations_cnn/reshape/strided_slice/stack?
8seq_seq_single_sations_cnn/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8seq_seq_single_sations_cnn/reshape/strided_slice/stack_1?
8seq_seq_single_sations_cnn/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8seq_seq_single_sations_cnn/reshape/strided_slice/stack_2?
0seq_seq_single_sations_cnn/reshape/strided_sliceStridedSlice1seq_seq_single_sations_cnn/reshape/Shape:output:0?seq_seq_single_sations_cnn/reshape/strided_slice/stack:output:0Aseq_seq_single_sations_cnn/reshape/strided_slice/stack_1:output:0Aseq_seq_single_sations_cnn/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0seq_seq_single_sations_cnn/reshape/strided_slice?
2seq_seq_single_sations_cnn/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :P24
2seq_seq_single_sations_cnn/reshape/Reshape/shape/1?
0seq_seq_single_sations_cnn/reshape/Reshape/shapePack9seq_seq_single_sations_cnn/reshape/strided_slice:output:0;seq_seq_single_sations_cnn/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0seq_seq_single_sations_cnn/reshape/Reshape/shape?
*seq_seq_single_sations_cnn/reshape/ReshapeReshape3seq_seq_single_sations_cnn/dense_1/BiasAdd:output:09seq_seq_single_sations_cnn/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P2,
*seq_seq_single_sations_cnn/reshape/Reshape?
IdentityIdentity3seq_seq_single_sations_cnn/reshape/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp9^seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOpE^seq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp8^seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp:^seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp:^seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp<^seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2t
8seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOp8seq_seq_single_sations_cnn/conv1d/BiasAdd/ReadVariableOp2?
Dseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOpDseq_seq_single_sations_cnn/conv1d/conv1d/ExpandDims_1/ReadVariableOp2r
7seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp7seq_seq_single_sations_cnn/dense/BiasAdd/ReadVariableOp2v
9seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp9seq_seq_single_sations_cnn/dense/Tensordot/ReadVariableOp2v
9seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp9seq_seq_single_sations_cnn/dense_1/BiasAdd/ReadVariableOp2z
;seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp;seq_seq_single_sations_cnn/dense_1/Tensordot/ReadVariableOp:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
?@
?
__inference__traced_save_8161
file_prefixG
Csavev2_seq_seq_single_sations_cnn_conv1d_kernel_read_readvariableopE
Asavev2_seq_seq_single_sations_cnn_conv1d_bias_read_readvariableopF
Bsavev2_seq_seq_single_sations_cnn_dense_kernel_read_readvariableopD
@savev2_seq_seq_single_sations_cnn_dense_bias_read_readvariableopH
Dsavev2_seq_seq_single_sations_cnn_dense_1_kernel_read_readvariableopF
Bsavev2_seq_seq_single_sations_cnn_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopN
Jsavev2_adam_seq_seq_single_sations_cnn_conv1d_kernel_m_read_readvariableopL
Hsavev2_adam_seq_seq_single_sations_cnn_conv1d_bias_m_read_readvariableopM
Isavev2_adam_seq_seq_single_sations_cnn_dense_kernel_m_read_readvariableopK
Gsavev2_adam_seq_seq_single_sations_cnn_dense_bias_m_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_dense_1_kernel_m_read_readvariableopM
Isavev2_adam_seq_seq_single_sations_cnn_dense_1_bias_m_read_readvariableopN
Jsavev2_adam_seq_seq_single_sations_cnn_conv1d_kernel_v_read_readvariableopL
Hsavev2_adam_seq_seq_single_sations_cnn_conv1d_bias_v_read_readvariableopM
Isavev2_adam_seq_seq_single_sations_cnn_dense_kernel_v_read_readvariableopK
Gsavev2_adam_seq_seq_single_sations_cnn_dense_bias_v_read_readvariableopO
Ksavev2_adam_seq_seq_single_sations_cnn_dense_1_kernel_v_read_readvariableopM
Isavev2_adam_seq_seq_single_sations_cnn_dense_1_bias_v_read_readvariableop
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
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Csavev2_seq_seq_single_sations_cnn_conv1d_kernel_read_readvariableopAsavev2_seq_seq_single_sations_cnn_conv1d_bias_read_readvariableopBsavev2_seq_seq_single_sations_cnn_dense_kernel_read_readvariableop@savev2_seq_seq_single_sations_cnn_dense_bias_read_readvariableopDsavev2_seq_seq_single_sations_cnn_dense_1_kernel_read_readvariableopBsavev2_seq_seq_single_sations_cnn_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopJsavev2_adam_seq_seq_single_sations_cnn_conv1d_kernel_m_read_readvariableopHsavev2_adam_seq_seq_single_sations_cnn_conv1d_bias_m_read_readvariableopIsavev2_adam_seq_seq_single_sations_cnn_dense_kernel_m_read_readvariableopGsavev2_adam_seq_seq_single_sations_cnn_dense_bias_m_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_dense_1_kernel_m_read_readvariableopIsavev2_adam_seq_seq_single_sations_cnn_dense_1_bias_m_read_readvariableopJsavev2_adam_seq_seq_single_sations_cnn_conv1d_kernel_v_read_readvariableopHsavev2_adam_seq_seq_single_sations_cnn_conv1d_bias_v_read_readvariableopIsavev2_adam_seq_seq_single_sations_cnn_dense_kernel_v_read_readvariableopGsavev2_adam_seq_seq_single_sations_cnn_dense_bias_v_read_readvariableopKsavev2_adam_seq_seq_single_sations_cnn_dense_1_kernel_v_read_readvariableopIsavev2_adam_seq_seq_single_sations_cnn_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: : : :  : : :: : : : : : : : : :  : : :: : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

:  : 
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
: : 

_output_shapes
: :$ 

_output_shapes

:  : 
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
: : 

_output_shapes
: :$ 

_output_shapes

:  : 
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
?
?
&__inference_dense_1_layer_call_fn_8008

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
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_74422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?!
?
?__inference_dense_layer_call_and_return_conditional_losses_7406

inputs3
!tensordot_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
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
:?????????P 2
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
:?????????P 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?	
?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7904
x
unknown: 
	unknown_0: 
	unknown_1:  
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
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_75692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????g

_user_specified_namex
?
_
C__inference_dropout_1_layer_call_and_return_conditional_losses_7560

inputs
identity^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
? 
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7683
input_1!
conv1d_7664: 
conv1d_7666: 

dense_7670:  

dense_7672: 
dense_1_7676: 
dense_1_7678:
identity??conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_7664conv1d_7666*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_73692 
conv1d/StatefulPartitionedCall?
dropout/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_75492
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_7670
dense_7672*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_74062
dense/StatefulPartitionedCall?
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_75602
dropout_1/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_7676dense_1_7678*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_74422!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_74602
reshape/PartitionedCall{
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
?
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7657
input_1!
conv1d_7640: 
conv1d_7642: 

dense_7645:  

dense_7647: 
dense_1_7650: 
dense_1_7652:
identity??conv1d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0conv1d_7640conv1d_7642*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_73692 
conv1d/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0
dense_7645
dense_7647*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_74062
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_7650dense_1_7652*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_74422!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_74602
reshape/PartitionedCall{
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
?
?
%__inference_conv1d_layer_call_fn_7929

inputs
unknown: 
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
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv1d_layer_call_and_return_conditional_losses_73692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????V: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????V
 
_user_specified_nameinputs
?h
?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7870
xH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: 9
'dense_tensordot_readvariableop_resource:  3
%dense_biasadd_readvariableop_resource: ;
)dense_1_tensordot_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:
identity??conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/Tensordot/ReadVariableOp
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
:?????????V*

begin_mask*
end_mask2
strided_slice?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDimsstrided_slice:output:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????V2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????P *
paddingVALID*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:?????????P *
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2
conv1d/Relu?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freew
dense/Tensordot/ShapeShapeconv1d/Relu:activations:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposeconv1d/Relu:activations:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P 2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P 2
dense/BiasAddn

dense/ReluReludense/BiasAdd:output:0*
T0*+
_output_shapes
:?????????P 2

dense/Relu?
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes?
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freez
dense_1/Tensordot/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape?
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis?
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2?
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis?
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const?
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod?
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1?
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1?
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis?
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat?
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack?
dense_1/Tensordot/transpose	Transposedense/Relu:activations:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????P 2
dense_1/Tensordot/transpose?
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_1/Tensordot/Reshape?
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/Tensordot/MatMul?
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/Const_2?
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis?
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1?
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????P2
dense_1/Tensordot?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????P2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :P2
reshape/Reshape/shape/1?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????P2
reshape/Reshapes
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identity?
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:N J
+
_output_shapes
:?????????g

_user_specified_namex
?	
?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7478
input_1
unknown: 
	unknown_0: 
	unknown_1:  
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
:?????????P*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_74632
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????P2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????g: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????g
!
_user_specified_name	input_1
?
?
$__inference_dense_layer_call_fn_7969

inputs
unknown:  
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
:?????????P *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_74062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????P : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_8017

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_7608

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P 2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P 2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P :S O
+
_output_shapes
:?????????P 
 
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
serving_default_input_1:0?????????g<
output_10
StatefulPartitionedCall:0?????????Ptensorflow/serving/predict:?|
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
>:< 2(seq_seq_single_sations_cnn/conv1d/kernel
4:2 2&seq_seq_single_sations_cnn/conv1d/bias
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
9:7  2'seq_seq_single_sations_cnn/dense/kernel
3:1 2%seq_seq_single_sations_cnn/dense/bias
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
;:9 2)seq_seq_single_sations_cnn/dense_1/kernel
5:32'seq_seq_single_sations_cnn/dense_1/bias
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
C:A 2/Adam/seq_seq_single_sations_cnn/conv1d/kernel/m
9:7 2-Adam/seq_seq_single_sations_cnn/conv1d/bias/m
>:<  2.Adam/seq_seq_single_sations_cnn/dense/kernel/m
8:6 2,Adam/seq_seq_single_sations_cnn/dense/bias/m
@:> 20Adam/seq_seq_single_sations_cnn/dense_1/kernel/m
::82.Adam/seq_seq_single_sations_cnn/dense_1/bias/m
C:A 2/Adam/seq_seq_single_sations_cnn/conv1d/kernel/v
9:7 2-Adam/seq_seq_single_sations_cnn/conv1d/bias/v
>:<  2.Adam/seq_seq_single_sations_cnn/dense/kernel/v
8:6 2,Adam/seq_seq_single_sations_cnn/dense/bias/v
@:> 20Adam/seq_seq_single_sations_cnn/dense_1/kernel/v
::82.Adam/seq_seq_single_sations_cnn/dense_1/bias/v
?2?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7789
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7870
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7657
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7683?
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
__inference__wrapped_model_7342input_1"?
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
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7478
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7887
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7904
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7633?
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
@__inference_conv1d_layer_call_and_return_conditional_losses_7920?
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
%__inference_conv1d_layer_call_fn_7929?
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
?__inference_dense_layer_call_and_return_conditional_losses_7960?
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
$__inference_dense_layer_call_fn_7969?
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
A__inference_dense_1_layer_call_and_return_conditional_losses_7999?
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
&__inference_dense_1_layer_call_fn_8008?
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
A__inference_dropout_layer_call_and_return_conditional_losses_8012
A__inference_dropout_layer_call_and_return_conditional_losses_8017?
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
&__inference_dropout_layer_call_fn_8022
&__inference_dropout_layer_call_fn_8027?
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
C__inference_dropout_1_layer_call_and_return_conditional_losses_8031
C__inference_dropout_1_layer_call_and_return_conditional_losses_8036?
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
(__inference_dropout_1_layer_call_fn_8041
(__inference_dropout_1_layer_call_fn_8046?
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
A__inference_reshape_layer_call_and_return_conditional_losses_8058?
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
&__inference_reshape_layer_call_fn_8063?
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
"__inference_signature_wrapper_7708input_1"?
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
__inference__wrapped_model_7342s4?1
*?'
%?"
input_1?????????g
? "3?0
.
output_1"?
output_1?????????P?
@__inference_conv1d_layer_call_and_return_conditional_losses_7920d3?0
)?&
$?!
inputs?????????V
? ")?&
?
0?????????P 
? ?
%__inference_conv1d_layer_call_fn_7929W3?0
)?&
$?!
inputs?????????V
? "??????????P ?
A__inference_dense_1_layer_call_and_return_conditional_losses_7999d3?0
)?&
$?!
inputs?????????P 
? ")?&
?
0?????????P
? ?
&__inference_dense_1_layer_call_fn_8008W3?0
)?&
$?!
inputs?????????P 
? "??????????P?
?__inference_dense_layer_call_and_return_conditional_losses_7960d3?0
)?&
$?!
inputs?????????P 
? ")?&
?
0?????????P 
? 
$__inference_dense_layer_call_fn_7969W3?0
)?&
$?!
inputs?????????P 
? "??????????P ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8031d7?4
-?*
$?!
inputs?????????P 
p
? ")?&
?
0?????????P 
? ?
C__inference_dropout_1_layer_call_and_return_conditional_losses_8036d7?4
-?*
$?!
inputs?????????P 
p 
? ")?&
?
0?????????P 
? ?
(__inference_dropout_1_layer_call_fn_8041W7?4
-?*
$?!
inputs?????????P 
p 
? "??????????P ?
(__inference_dropout_1_layer_call_fn_8046W7?4
-?*
$?!
inputs?????????P 
p
? "??????????P ?
A__inference_dropout_layer_call_and_return_conditional_losses_8012d7?4
-?*
$?!
inputs?????????P 
p
? ")?&
?
0?????????P 
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_8017d7?4
-?*
$?!
inputs?????????P 
p 
? ")?&
?
0?????????P 
? ?
&__inference_dropout_layer_call_fn_8022W7?4
-?*
$?!
inputs?????????P 
p 
? "??????????P ?
&__inference_dropout_layer_call_fn_8027W7?4
-?*
$?!
inputs?????????P 
p
? "??????????P ?
A__inference_reshape_layer_call_and_return_conditional_losses_8058\3?0
)?&
$?!
inputs?????????P
? "%?"
?
0?????????P
? y
&__inference_reshape_layer_call_fn_8063O3?0
)?&
$?!
inputs?????????P
? "??????????P?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7657i8?5
.?+
%?"
input_1?????????g
p 
? "%?"
?
0?????????P
? ?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7683i8?5
.?+
%?"
input_1?????????g
p
? "%?"
?
0?????????P
? ?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7789c2?/
(?%
?
x?????????g
p 
? "%?"
?
0?????????P
? ?
T__inference_seq_seq_single_sations_cnn_layer_call_and_return_conditional_losses_7870c2?/
(?%
?
x?????????g
p
? "%?"
?
0?????????P
? ?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7478\8?5
.?+
%?"
input_1?????????g
p 
? "??????????P?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7633\8?5
.?+
%?"
input_1?????????g
p
? "??????????P?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7887V2?/
(?%
?
x?????????g
p 
? "??????????P?
9__inference_seq_seq_single_sations_cnn_layer_call_fn_7904V2?/
(?%
?
x?????????g
p
? "??????????P?
"__inference_signature_wrapper_7708~??<
? 
5?2
0
input_1%?"
input_1?????????g"3?0
.
output_1"?
output_1?????????P