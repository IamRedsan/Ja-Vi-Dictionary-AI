╓Ъ,
╢Ж
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
б
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
.
Identity

input"T
output"T"	
Ttype
▄
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0■        "
value_indexint(0■        "+

vocab_sizeint         (0         "
	delimiterstring	"
offsetint И
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48аз+
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ь
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
д

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
А
Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
R
Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 R
         
Y
asset_path_initializer_2Placeholder*
_output_shapes
: *
dtype0*
shape: 
д

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
А
Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_2*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
R
Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 R
         
Y
asset_path_initializer_3Placeholder*
_output_shapes
: *
dtype0*
shape: 
д

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape: *
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
А
Variable_3/AssignAssignVariableOp
Variable_3asset_path_initializer_3*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
: *
dtype0
О

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*7
shared_name(&hash_table_/content/vocab.vi.txt_-2_-1*
value_dtype0	
Р
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*7
shared_name(&hash_table_/content/vocab.ja.txt_-2_-1*
value_dtype0	
Л

Variable_4VarHandleOp*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:░ъ*
shared_name
Variable_4
g
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes

:░ъ*
dtype0
Л

Variable_5VarHandleOp*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:░ъ*
shared_name
Variable_5
g
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes

:░ъ*
dtype0
e
ReadVariableOpReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
╞
StatefulPartitionedCallStatefulPartitionedCallReadVariableOphash_table_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__initializer_6114
c
ReadVariableOp_1ReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
╚
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOp_1
hash_table*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__initializer_6128
Р
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign
ё
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*к
valueаBЭ BЦ
$
ja
vi

signatures*
░
	tokenizer
_reserved_tokens
_vocab_path
	vocab

detokenize
	get_reserved_tokens

get_vocab_path
get_vocab_size

lookup
tokenize*
░
	tokenizer
_reserved_tokens
_vocab_path
	vocab

detokenize
get_reserved_tokens
get_vocab_path
get_vocab_size

lookup
tokenize*
* 
2
_basic_tokenizer
_wordpiece_tokenizer* 
* 
* 
GA
VARIABLE_VALUE
Variable_5#ja/vocab/.ATTRIBUTES/VARIABLE_VALUE*

trace_0
trace_1* 

trace_0* 

trace_0* 

trace_0* 

trace_0
 trace_1* 

!trace_0* 
2
"_basic_tokenizer
#_wordpiece_tokenizer* 
* 
* 
GA
VARIABLE_VALUE
Variable_4#vi/vocab/.ATTRIBUTES/VARIABLE_VALUE*

$trace_0
%trace_1* 

&trace_0* 

'trace_0* 

(trace_0* 

)trace_0
*trace_1* 

+trace_0* 
* 

,_vocab_lookup_table* 
* 
* 
* 

	capture_0* 
* 
* 
* 
/
-	capture_1
.	capture_2
/	capture_3* 
* 

0_vocab_lookup_table* 
* 
* 
* 

	capture_0* 
* 
* 
* 
/
1	capture_1
.	capture_2
/	capture_3* 
R
2_initializer
3_create_resource
4_initialize
5_destroy_resource* 
* 
* 
* 
R
6_initializer
7_create_resource
8_initialize
9_destroy_resource* 
* 

:	_filename* 

;trace_0* 

<trace_0* 

=trace_0* 

>	_filename* 

?trace_0* 

@trace_0* 

Atrace_0* 
* 
* 

:	capture_0* 
* 
* 
* 

>	capture_0* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╡
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
Variable_5
Variable_4Const_4*
Tin
2*
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
GPU 2J 8В *&
f!R
__inference__traced_save_6190
о
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename
Variable_5
Variable_4*
Tin
2*
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
GPU 2J 8В *)
f$R"
 __inference__traced_restore_6205ГЄ*
ў
║
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4292╨
╦raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╖
П
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_4330k
graggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all
h
draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1	f
braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = ▓
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = К
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_allNRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_3/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_3/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_3/assert_equal_1/All
─

В
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5496i
eraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all
E
Araggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder	G
Craggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
А
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 П
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityeraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all;^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╢
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "Н
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :^ Z

_output_shapes
: 
@
_user_specified_name(&RaggedFromRowSplits/assert_equal_1/All
з"
▀
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4166╬
╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
З
м
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5536в
Эraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All
З

о
__inference_lookup_3873
	token_ids	
token_ids_1	4
$raggedgather_readvariableop_resource:
░ъ
identity

identity_1	ИвRaggedGather/ReadVariableOp~
RaggedGather/ReadVariableOpReadVariableOp$raggedgather_readvariableop_resource*
_output_shapes

:░ъ*
dtype0\
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
RaggedGather/GatherV2GatherV2#RaggedGather/ReadVariableOp:value:0	token_ids#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         i
IdentityIdentityRaggedGather/GatherV2:output:0^NoOp*
T0*#
_output_shapes
:         X

Identity_1Identitytoken_ids_1^NoOp*
T0	*#
_output_shapes
:         @
NoOpNoOp^RaggedGather/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2:
RaggedGather/ReadVariableOpRaggedGather/ReadVariableOp:($
"
_user_specified_name
resource:NJ
#
_output_shapes
:         
#
_user_specified_name	token_ids:N J
#
_output_shapes
:         
#
_user_specified_name	token_ids
╧
4
$__inference_get_reserved_tokens_3842
identityp
ConstConst*
_output_shapes
:*
dtype0*7
value.B,B[PAD]B[UNK]B[CLS]B[SEP]B[MASK]I
IdentityIdentityConst:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
з"
▀
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4053╬
╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
ф

О
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5609m
iraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all
G
Craggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all=^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_1/assert_equal_1/All
Р
╙
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_4486Y
Uraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all
g
craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	X
Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_sub	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
Ив5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert┐
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 1 (=y) have incompatible shapes.и
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:│
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = д
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*8
value/B- B'y (RaggedConcat/RaggedNRows_1/sub:0) = ╝
5RaggedConcat/assert_equal_1/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_allERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_sub*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all6^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ▐
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:04^RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Й
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOpNoOp6^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert:VR

_output_shapes
: 
8
_user_specified_name RaggedConcat/RaggedNRows_1/sub:ea

_output_shapes
: 
G
_user_specified_name/-RaggedConcat/RaggedFromTensor/strided_slice_4:W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_1/All
ч
а
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5423Ю
Щraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
_
[raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	a
]raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ъ
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 °
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЩraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allU^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ъ
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┴
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :x t

_output_shapes
: 
Z
_user_specified_nameB@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All
Р
╙
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_6006Y
Uraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all
g
craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	X
Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_sub	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
Ив5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert┐
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 1 (=y) have incompatible shapes.и
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:│
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = д
<RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*8
value/B- B'y (RaggedConcat/RaggedNRows_1/sub:0) = ╝
5RaggedConcat/assert_equal_1/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_allERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0craggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4ERaggedConcat/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Traggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_raggednrows_1_sub*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_1_assert_assertguard_assert_raggedconcat_assert_equal_1_all6^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ▐
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:04^RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Й
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOpNoOp6^RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_1/Assert/AssertGuard/Assert:VR

_output_shapes
: 
8
_user_specified_name RaggedConcat/RaggedNRows_1/sub:ea

_output_shapes
: 
G
_user_specified_name/-RaggedConcat/RaggedFromTensor/strided_slice_4:W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_1/All
№
<
__inference_get_vocab_path_3847
unknown
identity>
IdentityIdentityunknown*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
у!
╤
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5460╩
┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
д
Яraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвmRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert 
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  р
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:я
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = л
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0Яraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ╘
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: Ж
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0l^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ∙
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpn^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "я
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2▐
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertmRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:vr
#
_output_shapes
:         
K
_user_specified_name31RaggedFromRowSplits/RowPartitionFromRowSplits/sub:Р Л

_output_shapes
: 
q
_user_specified_nameYWRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
Я
╢
__inference__traced_save_6190
file_prefix1
!read_disablecopyonread_variable_5:
░ъ3
#read_1_disablecopyonread_variable_4:
░ъ
savev2_const_4

identity_5ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: s
Read/DisableCopyOnReadDisableCopyOnRead!read_disablecopyonread_variable_5"/device:CPU:0*
_output_shapes
 Ы
Read/ReadVariableOpReadVariableOp!read_disablecopyonread_variable_5^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:░ъ*
dtype0g
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:░ъ_

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:░ъw
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_variable_4"/device:CPU:0*
_output_shapes
 б
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_variable_4^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:░ъ*
dtype0k

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:░ъa

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:░ъ╤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*{
valuerBpB#ja/vocab/.ATTRIBUTES/VARIABLE_VALUEB#vi/vocab/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B Ж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0savev2_const_4"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 h

Identity_4Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: S

Identity_5IdentityIdentity_4:output:0^NoOp*
T0*
_output_shapes
: Щ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp*
_output_shapes
 "!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp:?;

_output_shapes
: 
!
_user_specified_name	Const_4:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╥	
╪
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_6035[
Wraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all
>
:raggedconcat_assert_equal_3_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_3_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
y
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all4^RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: и
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_3/All
З
м
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5776в
Эraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All
▐"
ї
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4257а
Ыraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All
ЮГ
б
__inference_detokenize_3471
	tokenized	0
,none_export_lookuptableexportv2_table_handle
identityИвAssert/AssertвNone_Export/LookupTableExportV2Ю
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::К
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:         *
shape:         О
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:         *
shape:         W
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
         X
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:         O
argsort/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Ri
argsort/subSubargsort/Neg:y:0argsort/sub/y:output:0*
T0	*#
_output_shapes
:         Z
argsort/ShapeShapeargsort/sub:z:0*
T0	*
_output_shapes
::э╧n
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :Ж
argsort/TopKV2TopKV2argsort/sub:z:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:         :         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:         Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▒

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╘
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskl
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:         K
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R[
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:         O
ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: B
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: ╣
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`┴
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`Й
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 R
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: b
MinimumMinimum	tokenizedCast:y:0*
T0	*0
_output_shapes
:                  ]
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : м

GatherV2_2GatherV2concat:output:0Minimum:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  w
RaggedFromTensor/ShapeShapeGatherV2_2:output:0*
T0*
_output_shapes
:*
out_type0	:э╨n
$RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
RaggedFromTensor/strided_sliceStridedSliceRaggedFromTensor/Shape:output:0-RaggedFromTensor/strided_slice/stack:output:0/RaggedFromTensor/strided_slice/stack_1:output:0/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_1StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_1/stack:output:01RaggedFromTensor/strided_slice_1/stack_1:output:01RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_2StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_2/stack:output:01RaggedFromTensor/strided_slice_2/stack_1:output:01RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskТ
RaggedFromTensor/mulMul)RaggedFromTensor/strided_slice_1:output:0)RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: p
&RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
 RaggedFromTensor/strided_slice_3StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_3/stack:output:01RaggedFromTensor/strided_slice_3/stack_1:output:01RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskp
 RaggedFromTensor/concat/values_0PackRaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:^
RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╬
RaggedFromTensor/concatConcatV2)RaggedFromTensor/concat/values_0:output:0)RaggedFromTensor/strided_slice_3:output:0%RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:Ц
RaggedFromTensor/ReshapeReshapeGatherV2_2:output:0 RaggedFromTensor/concat:output:0*
Tshape0	*
T0*#
_output_shapes
:         p
&RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_4StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_4/stack:output:01RaggedFromTensor/strided_slice_4/stack_1:output:01RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_5StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_5/stack:output:01RaggedFromTensor/strided_slice_5/stack_1:output:01RaggedFromTensor/strided_slice_5/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskа
1RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape!RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	:э╨Й
?RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlice:RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0HRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskФ
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RВ
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2)RaggedFromTensor/strided_slice_4:output:0[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: Ъ
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R Ъ
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 Rи
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangeaRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0TRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0aRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         Н
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0)RaggedFromTensor/strided_slice_5:output:0*
T0	*#
_output_shapes
:         й
RaggedSegmentJoin/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
::э╧o
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :Й
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: И
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskК
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Х
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╥
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskы
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         ╥
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
:*
out_type0	:э╨У
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskt
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R┌
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :о
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ░
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ш
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         п
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         о
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧П
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Д
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Б
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: ч
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ц
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximum?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zero:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: Ж
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : И
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :╤
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ш
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         й
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         р
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         Ю
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :В
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Д
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:Л
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┴
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ь
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Я
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: г
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╥
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:М
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Р
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         б
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         г
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ╔
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ф
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
О
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         У
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoin!RaggedFromTensor/Reshape:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator а
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:         *
pattern \#\#*
rewrite С
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:         *
pattern	^ +| +$*
rewrite S
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B г
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:         :         :p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskл
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         в
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ╝
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/SizeSizeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
: Т
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : л
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Size:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ╫
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Ц
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: С
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: О
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :а
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: Ф
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: з
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         о
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:         С
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 е
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincountDenseBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*

Tidx0*
T0	*#
_output_shapes
:         Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : п
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincount:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Ч
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         д
StaticRegexFullMatchStaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:         *-
pattern" \[PAD\]|\[CLS\]|\[SEP\]|\[MASK\]\

LogicalNot
LogicalNotStaticRegexFullMatch:output:0*#
_output_shapes
:         b
RaggedMask/assert_equal/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
RaggedMask/CastCastLogicalNot:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:         ╩
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
::э╧Ч
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :д
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: ░
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:х
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╜
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:я
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЖ
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         є
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	:э╨╗
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▌
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЬ
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 Rї
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :└
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┬
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: ╝
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ┴
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         └
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧╖
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:°
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Я
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         й
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: В
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zeroConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : Б
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zero:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: о
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ░
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :ї
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         └
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         ─
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Є
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         ╣
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  к
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         м
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:ж
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ╙
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧─
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:е
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:╟
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╛
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:┐
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╜
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskф
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:┤
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : М
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:л
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         ╔
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ╛
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         █
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         Ў
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
╢
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ┴
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Х
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
Tindices0	*
T0	*#
_output_shapes
:         w
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : в
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Г
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R А
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         │
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         з
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧Я
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:ь
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:в
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╧
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: й
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Ж
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskй
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Д
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskЪ
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:П
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ╙
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:╗
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:         д
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ▓
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         С
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         м
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
С
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : н
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         q
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
::э╧q
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :П
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: К
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskМ
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskё
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Ъ
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Х
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskv
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rр
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┤
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: а
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         │
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         ▓
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧С
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╒
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:К
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Г
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: э
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ь
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zero:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: И
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :┘
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ъ
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         п
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         ф
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         д
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  Д
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :И
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Ж
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :И
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:С
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┼
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:б
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: й
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╓
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:О
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : щ
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Ц
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         г
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         й
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ═
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ш
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
Р
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         м
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator {
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^NoOp*
T0*#
_output_shapes
:         T
NoOpNoOp^Assert/Assert ^None_Export/LookupTableExportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:                  : 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:,(
&
_user_specified_nametable_handle:[ W
0
_output_shapes
:                  
#
_user_specified_name	tokenized
ў
║
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4165╨
╦raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╧╔
▓
__inference_detokenize_5358
	tokenized	
tokenized_1	0
,none_export_lookuptableexportv2_table_handle
identityИвAssert/AssertвNone_Export/LookupTableExportV2Ю
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::К
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:         *
shape:         О
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:         *
shape:         W
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
         X
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:         O
argsort/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Ri
argsort/subSubargsort/Neg:y:0argsort/sub/y:output:0*
T0	*#
_output_shapes
:         Z
argsort/ShapeShapeargsort/sub:z:0*
T0	*
_output_shapes
::э╧n
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :Ж
argsort/TopKV2TopKV2argsort/sub:z:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:         :         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:         Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▒

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╘
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskl
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:         K
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R[
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:         O
ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: B
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: ╣
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`┴
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`Й
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 R
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: U
MinimumMinimum	tokenizedCast:y:0*
T0	*#
_output_shapes
:         ]
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         \
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╡
RaggedGather/GatherV2GatherV2concat:output:0Minimum:z:0#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         `
RaggedSegmentJoin/ShapeShapetokenized_1*
T0	*
_output_shapes
::э╧o
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :Й
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: И
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSlicetokenized_1GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskК
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Х
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSlicetokenized_1IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskы
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Й
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapetokenized_1*
T0	*
_output_shapes
:*
out_type0	:э╨У
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskt
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R┌
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :о
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ░
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ш
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         п
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         о
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧П
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Д
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Б
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: ч
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ц
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximum?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zero:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: Ж
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : И
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :╤
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ш
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         й
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         р
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         Ю
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :В
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Д
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:Л
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┴
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ь
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Я
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: г
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╥
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:М
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Р
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         б
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         г
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ╔
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ф
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
О
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Р
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoinRaggedGather/GatherV2:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator а
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:         *
pattern \#\#*
rewrite С
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:         *
pattern	^ +| +$*
rewrite S
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B г
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:         :         :p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskл
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         в
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ╝
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/SizeSizeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
: Т
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : л
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Size:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ╫
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Ц
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: С
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: О
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :а
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: Ф
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: з
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         о
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:         С
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 е
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincountDenseBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*

Tidx0*
T0	*#
_output_shapes
:         Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : п
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincount:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Ч
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         д
StaticRegexFullMatchStaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:         *-
pattern" \[PAD\]|\[CLS\]|\[SEP\]|\[MASK\]\

LogicalNot
LogicalNotStaticRegexFullMatch:output:0*#
_output_shapes
:         b
RaggedMask/assert_equal/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
RaggedMask/CastCastLogicalNot:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:         ╩
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
::э╧Ч
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :д
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: ░
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:х
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╜
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:я
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЖ
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         є
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	:э╨╗
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▌
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЬ
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 Rї
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :└
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┬
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: ╝
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ┴
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         └
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧╖
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:°
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Я
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         й
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: В
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zeroConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : Б
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zero:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: о
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ░
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :ї
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         └
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         ─
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Є
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         ╣
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  к
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         м
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:ж
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ╙
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧─
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:е
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:╟
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╛
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:┐
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╜
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskф
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:┤
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : М
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:л
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         ╔
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ╛
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         █
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         Ў
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
╢
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ┴
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Х
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
Tindices0	*
T0	*#
_output_shapes
:         w
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : в
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Г
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R А
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         │
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         з
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧Я
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:ь
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:в
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╧
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: й
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Ж
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskй
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Д
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskЪ
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:П
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ╙
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:╗
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:         д
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ▓
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         С
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         м
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
С
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : н
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         q
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
::э╧q
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :П
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: К
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskМ
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskё
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Ъ
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Х
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskv
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rр
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┤
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: а
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         │
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         ▓
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧С
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╒
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:К
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Г
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: э
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ь
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zero:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: И
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :┘
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ъ
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         п
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         ф
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         д
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  Д
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :И
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Ж
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :И
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:С
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┼
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:б
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: й
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╓
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:О
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : щ
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Ц
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         г
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         й
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ═
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ш
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
Р
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         м
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator {
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^NoOp*
T0*#
_output_shapes
:         T
NoOpNoOp^Assert/Assert ^None_Export/LookupTableExportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:,(
&
_user_specified_nametable_handle:NJ
#
_output_shapes
:         
#
_user_specified_name	tokenized:N J
#
_output_shapes
:         
#
_user_specified_name	tokenized
Ў
a
__inference_get_vocab_size_5377-
shape_readvariableop_resource:
░ъ
identityИp
Shape/ReadVariableOpReadVariableOpshape_readvariableop_resource*
_output_shapes

:░ъ*
dtype0Q
ShapeConst*
_output_shapes
:*
dtype0*
valueB:░ъ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :( $
"
_user_specified_name
resource
▐"
ї
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4017а
Ыraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All
з"
▀
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5813╬
╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╖
П
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_5723k
graggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all
h
draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1	f
braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = ▓
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = К
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_allNRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_2/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_2/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_2/assert_equal_1/All
Щ
+
__inference__destroyer_6118
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ф

О
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_4202m
iraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all
G
Craggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all=^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_2/assert_equal_1/All
Ц
c
__inference_lookup_3863
	token_ids	
gather_resource:
░ъ
identityИвGatherГ
GatherResourceGathergather_resource	token_ids*
Tindices0	*0
_output_shapes
:                  *
dtype0g
IdentityIdentityGather:output:0^NoOp*
T0*0
_output_shapes
:                  +
NoOpNoOp^Gather*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:                  : 2
GatherGather:($
"
_user_specified_name
resource:[ W
0
_output_shapes
:                  
#
_user_specified_name	token_ids
╥	
╪
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_6005[
Wraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all
>
:raggedconcat_assert_equal_1_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_1_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
y
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all4^RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: и
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_1/All
К"
у
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3904Ь
Чraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
Ч
Тraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slice	П
Кraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвVRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertр
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╔
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:т
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = ┌
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = Л
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЧraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Тraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slicefRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Кraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 °
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЧraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ┴
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0U^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╦
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpW^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┴
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2░
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertVRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:kg

_output_shapes
: 
M
_user_specified_name53RaggedFromRowSplits/RowPartitionFromRowSplits/Const:so

_output_shapes
: 
U
_user_specified_name=;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:x t

_output_shapes
: 
Z
_user_specified_nameB@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All
З
м
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4256в
Эraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All
з"
▀
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5686╬
╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
┘
░
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5459╠
╟raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
v
rraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
▒
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ╘
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╟raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alll^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ш
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "я
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Р Л

_output_shapes
: 
q
_user_specified_nameYWRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╧╔
▓
__inference_detokenize_3838
	tokenized	
tokenized_1	0
,none_export_lookuptableexportv2_table_handle
identityИвAssert/AssertвNone_Export/LookupTableExportV2Ю
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::К
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:         *
shape:         О
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:         *
shape:         W
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
         X
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:         O
argsort/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Ri
argsort/subSubargsort/Neg:y:0argsort/sub/y:output:0*
T0	*#
_output_shapes
:         Z
argsort/ShapeShapeargsort/sub:z:0*
T0	*
_output_shapes
::э╧n
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :Ж
argsort/TopKV2TopKV2argsort/sub:z:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:         :         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:         Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▒

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╘
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskl
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:         K
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R[
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:         O
ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: B
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: ╣
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`┴
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`Й
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 R
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: U
MinimumMinimum	tokenizedCast:y:0*
T0	*#
_output_shapes
:         ]
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         \
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╡
RaggedGather/GatherV2GatherV2concat:output:0Minimum:z:0#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         `
RaggedSegmentJoin/ShapeShapetokenized_1*
T0	*
_output_shapes
::э╧o
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :Й
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: И
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB: 
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSlicetokenized_1GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskК
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Х
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Й
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSlicetokenized_1IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskы
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Й
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapetokenized_1*
T0	*
_output_shapes
:*
out_type0	:э╨У
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskt
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R┌
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :о
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ░
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ш
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         п
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         о
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧П
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Д
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Б
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: ч
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ц
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximum?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zero:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: Ж
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : И
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :╤
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ш
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         й
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         р
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         Ю
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :В
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Д
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:Л
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┴
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ь
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Я
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: г
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╥
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:М
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Р
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         б
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         г
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ╔
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ф
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
О
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Р
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoinRaggedGather/GatherV2:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator а
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:         *
pattern \#\#*
rewrite С
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:         *
pattern	^ +| +$*
rewrite S
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B г
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:         :         :p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskл
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         в
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ╝
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/SizeSizeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
: Т
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : л
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Size:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ╫
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Ц
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: С
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: О
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :а
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: Ф
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: з
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         о
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:         С
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 е
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincountDenseBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*

Tidx0*
T0	*#
_output_shapes
:         Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : п
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincount:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Ч
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         д
StaticRegexFullMatchStaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:         *-
pattern" \[PAD\]|\[CLS\]|\[SEP\]|\[MASK\]\

LogicalNot
LogicalNotStaticRegexFullMatch:output:0*#
_output_shapes
:         b
RaggedMask/assert_equal/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
RaggedMask/CastCastLogicalNot:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:         ╩
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
::э╧Ч
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :д
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: ░
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:х
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╜
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:я
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЖ
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         є
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	:э╨╗
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▌
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЬ
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 Rї
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :└
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┬
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: ╝
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ┴
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         └
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧╖
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:°
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Я
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         й
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: В
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zeroConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : Б
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zero:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: о
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ░
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :ї
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         └
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         ─
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Є
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         ╣
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  к
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         м
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:ж
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ╙
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧─
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:е
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:╟
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╛
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:┐
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╜
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskф
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:┤
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : М
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:л
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         ╔
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ╛
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         █
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         Ў
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
╢
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ┴
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Х
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
Tindices0	*
T0	*#
_output_shapes
:         w
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : в
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Г
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R А
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         │
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         з
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧Я
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:ь
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:в
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╧
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: й
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Ж
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskй
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Д
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskЪ
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:П
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ╙
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:╗
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:         д
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ▓
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         С
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         м
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
С
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : н
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         q
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
::э╧q
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :П
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: К
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskМ
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskё
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Ъ
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Х
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskv
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rр
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┤
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: а
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         │
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         ▓
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧С
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╒
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:К
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Г
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: э
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ь
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zero:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: И
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :┘
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ъ
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         п
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         ф
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         д
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  Д
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :И
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Ж
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :И
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:С
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┼
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:б
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: й
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╓
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:О
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : щ
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Ц
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         г
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         й
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ═
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ш
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
Р
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         м
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator {
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^NoOp*
T0*#
_output_shapes
:         T
NoOpNoOp^Assert/Assert ^None_Export/LookupTableExportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:,(
&
_user_specified_nametable_handle:NJ
#
_output_shapes
:         
#
_user_specified_name	tokenized:N J
#
_output_shapes
:         
#
_user_specified_name	tokenized
╠
9
__inference__creator_6108
identityИв
hash_tableО

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*7
shared_name(&hash_table_/content/vocab.ja.txt_-2_-1*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
т
¤
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3977g
craggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all
d
`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1	b
^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ив<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert┼
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorп
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:░
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = о
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = Ї
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertcraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_allLRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 П
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentitycraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: є
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0;^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ч
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "Н
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2|
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:YU

_output_shapes
: 
;
_user_specified_name#!RaggedFromRowSplits/strided_slice:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits/strided_slice_1:^ Z

_output_shapes
: 
@
_user_specified_name(&RaggedFromRowSplits/assert_equal_1/All
╩
╩
 __inference__traced_restore_6205
file_prefix+
assignvariableop_variable_5:
░ъ-
assignvariableop_1_variable_4:
░ъ

identity_3ИвAssignVariableOpвAssignVariableOp_1╘
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*{
valuerBpB#ja/vocab/.ATTRIBUTES/VARIABLE_VALUEB#vi/vocab/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B н
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:о
AssignVariableOpAssignVariableOpassignvariableop_variable_5Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_4Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 В

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: L
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2(
AssignVariableOp_1AssignVariableOp_12$
AssignVariableOpAssignVariableOp:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
┘
░
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_3939╠
╟raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
v
rraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
▒
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ╘
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╟raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alll^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ш
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "я
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Р Л

_output_shapes
: 
q
_user_specified_nameYWRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
з"
▀
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4293╬
╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╥	
╪
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_4485[
Wraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all
>
:raggedconcat_assert_equal_1_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_1_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_1_assert_assertguard_identity_1
y
3RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_1_assert_assertguard_identity_raggedconcat_assert_equal_1_all4^RaggedConcat/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: и
9RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
9raggedconcat_assert_equal_1_assert_assertguard_identity_1BRaggedConcat/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_1/All
─

В
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3976i
eraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all
E
Araggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder	G
Craggedfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
А
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 П
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityeraggedfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_assert_equal_1_all;^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╢
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "Н
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :^ Z

_output_shapes
: 
@
_user_specified_name(&RaggedFromRowSplits/assert_equal_1/All
▐"
ї
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5537а
Ыraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All
З
м
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4129в
Эraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All
Б
▄
__inference__initializer_6128*
&text_file_id_table_init_asset_filepathF
Btext_file_id_table_init_initializetablefromtextfilev2_table_handle
identityИв5text_file_id_table_init/InitializeTableFromTextFileV2О
5text_file_id_table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2Btext_file_id_table_init_initializetablefromtextfilev2_table_handle&text_file_id_table_init_asset_filepath*
_output_shapes
 *
	key_index■        *
value_index         G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: Z
NoOpNoOp6^text_file_id_table_init/InitializeTableFromTextFileV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2n
5text_file_id_table_init/InitializeTableFromTextFileV25text_file_id_table_init/InitializeTableFromTextFileV2:,(
&
_user_specified_nametable_handle: 

_output_shapes
: 
ф

О
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_5849m
iraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all
G
Craggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all=^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_3/assert_equal_1/All
у!
╤
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_3940╩
┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
д
Яraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub	u
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвmRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert 
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  р
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:я
tRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = л
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0}RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0Яraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ╘
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity┼raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: Ж
qRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityxRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0l^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ∙
kRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpn^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "я
qraggedfromrowsplits_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1zRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2▐
mRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertmRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:vr
#
_output_shapes
:         
K
_user_specified_name31RaggedFromRowSplits/RowPartitionFromRowSplits/sub:Р Л

_output_shapes
: 
q
_user_specified_nameYWRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╖
П
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_4203k
graggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all
h
draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1	f
braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = ▓
ERaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = К
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_allNRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice_1NRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_2_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_assert_equal_1_all?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_2/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_2/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_2/assert_equal_1/All
ф

О
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_4329m
iraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all
G
Craggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_3_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_3_assert_equal_1_all=^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_3/assert_equal_1/All
▐"
ї
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5777а
Ыraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All
ў
║
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5572╨
╦raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
ч
а
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3903Ю
Щraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
_
[raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	a
]raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ъ
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 °
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЩraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allU^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ъ
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┴
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :x t

_output_shapes
: 
Z
_user_specified_nameB@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All
╧
4
$__inference_get_reserved_tokens_5362
identityp
ConstConst*
_output_shapes
:*
dtype0*7
value.B,B[PAD]B[UNK]B[CLS]B[SEP]B[MASK]I
IdentityIdentityConst:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
╠
9
__inference__creator_6122
identityИв
hash_tableО

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*7
shared_name(&hash_table_/content/vocab.vi.txt_-2_-1*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
▐"
ї
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5650а
Ыraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All
т
¤
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5497g
craggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all
d
`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1	b
^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice	D
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ив<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert┼
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorп
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:░
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = о
CRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = Ї
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertcraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_allLRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0`raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice_1LRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0^raggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 П
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentitycraggedfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_assert_equal_1_all=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: є
@RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityGRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0;^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ч
:RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp=^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "Н
@raggedfromrowsplits_assert_equal_1_assert_assertguard_identity_1IRaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2|
<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert<RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:YU

_output_shapes
: 
;
_user_specified_name#!RaggedFromRowSplits/strided_slice:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits/strided_slice_1:^ Z

_output_shapes
: 
@
_user_specified_name(&RaggedFromRowSplits/assert_equal_1/All
ў
║
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4052╨
╦raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
╖
П
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5610k
graggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all
h
draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1	f
braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = ▓
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = К
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_allNRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_1/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_1/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_1/assert_equal_1/All
╘
ф
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_4516Y
Uraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all
g
craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	i
eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
Ив5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert┐
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 2 (=y) have incompatible shapes.и
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:│
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = ╡
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*I
value@B> B8y (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = ═
5RaggedConcat/assert_equal_3/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_allERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2:output:0craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4:output:0eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all6^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ▐
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:04^RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Й
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOpNoOp6^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
_output_shapes
 "
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert:gc

_output_shapes
: 
I
_user_specified_name1/RaggedConcat/RaggedFromTensor_1/strided_slice_4:ea

_output_shapes
: 
G
_user_specified_name/-RaggedConcat/RaggedFromTensor/strided_slice_4:W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_3/All
▐"
ї
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4130а
Ыraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
Ы
Цraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slice	У
Оraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвXRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertт
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╦
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:ц
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = ▐
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = б
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЫraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Цraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_strided_slicehRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Оraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЫraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allY^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ╟
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0W^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╧
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpY^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┼
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2┤
XRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertXRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:mi

_output_shapes
: 
O
_user_specified_name75RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:uq

_output_shapes
: 
W
_user_specified_name?=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All
Щ
+
__inference__destroyer_6132
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ф

О
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_5722m
iraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all
G
Craggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_2_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_assert_equal_1_all=^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_2_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_2/assert_equal_1/All
К"
у
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5424Ь
Чraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_all
Ч
Тraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slice	П
Кraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const	^
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
ИвVRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assertр
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╔
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:т
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = ┌
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = Л
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertAssertЧraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0Тraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_strided_slicefRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0Кraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_const*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 °
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЧraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ┴
ZRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentityaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0U^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ╦
TRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOpW^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "┴
Zraggedfromrowsplits_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1cRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2░
VRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/AssertVRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Assert:kg

_output_shapes
: 
M
_user_specified_name53RaggedFromRowSplits/RowPartitionFromRowSplits/Const:so

_output_shapes
: 
U
_user_specified_name=;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:x t

_output_shapes
: 
Z
_user_specified_nameB@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All
ЮГ
б
__inference_detokenize_4991
	tokenized	0
,none_export_lookuptableexportv2_table_handle
identityИвAssert/AssertвNone_Export/LookupTableExportV2Ю
None_Export/LookupTableExportV2LookupTableExportV2,none_export_lookuptableexportv2_table_handle*
Tkeys0*
Tvalues0	*
_output_shapes

::К
EnsureShapeEnsureShape&None_Export/LookupTableExportV2:keys:0*
T0*#
_output_shapes
:         *
shape:         О
EnsureShape_1EnsureShape(None_Export/LookupTableExportV2:values:0*
T0	*#
_output_shapes
:         *
shape:         W
argsort/axisConst*
_output_shapes
: *
dtype0*
valueB :
         X
argsort/NegNegEnsureShape_1:output:0*
T0	*#
_output_shapes
:         O
argsort/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Ri
argsort/subSubargsort/Neg:y:0argsort/sub/y:output:0*
T0	*#
_output_shapes
:         Z
argsort/ShapeShapeargsort/sub:z:0*
T0	*
_output_shapes
::э╧n
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         g
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskN
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :Ж
argsort/TopKV2TopKV2argsort/sub:z:0argsort/strided_slice:output:0*
T0	*2
_output_shapes 
:         :         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : п
GatherV2GatherV2EnsureShape_1:output:0argsort/TopKV2:indices:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*#
_output_shapes
:         Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▒

GatherV2_1GatherV2EnsureShape:output:0argsort/TopKV2:indices:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:         ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╘
strided_sliceStridedSliceGatherV2:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskI
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Y
EqualEqualstrided_slice:output:0Equal/y:output:0*
T0	*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
strided_slice_1StridedSliceGatherV2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: j
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:у
strided_slice_2StridedSliceGatherV2:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskl
subSubstrided_slice_1:output:0strided_slice_2:output:0*
T0	*#
_output_shapes
:         K
	Equal_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R[
Equal_1Equalsub:z:0Equal_1/y:output:0*
T0	*#
_output_shapes
:         O
ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
AllAllEqual_1:z:0Const:output:0*
_output_shapes
: B
and
LogicalAnd	Equal:z:0All:output:0*
_output_shapes
: ╣
Assert/ConstConst*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`┴
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*}
valuetBr Bl`detokenize` only works with vocabulary tables where the indices are dense on the interval `[0, vocab_size)`Й
Assert/AssertAssertand:z:0Assert/Assert/data_0:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 R
SizeSizeGatherV2_1:output:0^Assert/Assert*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0	*

SrcT0*
_output_shapes
: b
MinimumMinimum	tokenizedCast:y:0*
T0	*0
_output_shapes
:                  ]
concat/values_1Const*
_output_shapes
:*
dtype0*
valueBB[UNK]M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : О
concatConcatV2GatherV2_1:output:0concat/values_1:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : м

GatherV2_2GatherV2concat:output:0Minimum:z:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:                  w
RaggedFromTensor/ShapeShapeGatherV2_2:output:0*
T0*
_output_shapes
:*
out_type0	:э╨n
$RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
RaggedFromTensor/strided_sliceStridedSliceRaggedFromTensor/Shape:output:0-RaggedFromTensor/strided_slice/stack:output:0/RaggedFromTensor/strided_slice/stack_1:output:0/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_1StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_1/stack:output:01RaggedFromTensor/strided_slice_1/stack_1:output:01RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_2StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_2/stack:output:01RaggedFromTensor/strided_slice_2/stack_1:output:01RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskТ
RaggedFromTensor/mulMul)RaggedFromTensor/strided_slice_1:output:0)RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: p
&RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
 RaggedFromTensor/strided_slice_3StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_3/stack:output:01RaggedFromTensor/strided_slice_3/stack_1:output:01RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskp
 RaggedFromTensor/concat/values_0PackRaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:^
RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╬
RaggedFromTensor/concatConcatV2)RaggedFromTensor/concat/values_0:output:0)RaggedFromTensor/strided_slice_3:output:0%RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:Ц
RaggedFromTensor/ReshapeReshapeGatherV2_2:output:0 RaggedFromTensor/concat:output:0*
Tshape0	*
T0*#
_output_shapes
:         p
&RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_4StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_4/stack:output:01RaggedFromTensor/strided_slice_4/stack_1:output:01RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskp
&RaggedFromTensor/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(RaggedFromTensor/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 RaggedFromTensor/strided_slice_5StridedSliceRaggedFromTensor/Shape:output:0/RaggedFromTensor/strided_slice_5/stack:output:01RaggedFromTensor/strided_slice_5/stack_1:output:01RaggedFromTensor/strided_slice_5/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskа
1RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape!RaggedFromTensor/Reshape:output:0*
T0*
_output_shapes
:*
out_type0	:э╨Й
?RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Л
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Л
ARaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
9RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSlice:RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0HRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0JRaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskФ
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RВ
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV2)RaggedFromTensor/strided_slice_4:output:0[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: Ъ
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R Ъ
XRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 Rи
RRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangeaRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0TRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0aRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         Н
PRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMul[RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0)RaggedFromTensor/strided_slice_5:output:0*
T0	*#
_output_shapes
:         й
RaggedSegmentJoin/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
::э╧o
%RaggedSegmentJoin/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedSegmentJoin/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedSegmentJoin/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedSegmentJoin/strided_sliceStridedSlice RaggedSegmentJoin/Shape:output:0.RaggedSegmentJoin/strided_slice/stack:output:00RaggedSegmentJoin/strided_slice/stack_1:output:00RaggedSegmentJoin/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
RaggedSegmentJoin/sub/yConst*
_output_shapes
: *
dtype0*
value	B :Й
RaggedSegmentJoin/subSub(RaggedSegmentJoin/strided_slice:output:0 RaggedSegmentJoin/sub/y:output:0*
T0*
_output_shapes
: И
>RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: К
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╚
8RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_sliceStridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskК
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Х
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╥
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskы
.RaggedSegmentJoin/RaggedSplitsToSegmentIds/subSubARaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice:output:0CRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         ╥
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/ShapeShapeTRaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0*
T0	*
_output_shapes
:*
out_type0	:э╨У
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
:RaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Shape:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0KRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskt
2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 R┌
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1SubCRaggedSegmentJoin/RaggedSplitsToSegmentIds/strided_slice_2:output:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :о
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/CastCast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ░
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1Cast?RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ш
0RaggedSegmentJoin/RaggedSplitsToSegmentIds/rangeRange9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast:y:04RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub_1:z:0;RaggedSegmentJoin/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         п
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/CastCast2RaggedSegmentJoin/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         о
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ShapeShape9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧П
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:С
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSlice@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Shape:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask├
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackHRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Д
=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo:RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Cast:y:0LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Б
7RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: ч
5RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaxMaxFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: x
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ц
9RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/MaximumMaximum?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/zero:output:0>RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: Ж
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : И
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :╤
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ш
MRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         й
IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsFRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         р
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastRRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         Ю
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :В
<RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims9RaggedSegmentJoin/RaggedSplitsToSegmentIds/range:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Д
BRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :В
@RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackKRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0=RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:Л
6RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/TileTileERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0IRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┴
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ь
RRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceMRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0[RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:Я
URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: г
CRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdURaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0^RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask├
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2Shape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0]RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0_RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╥
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:М
JRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
ERaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0SRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Р
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshape?RaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/Tile:output:0NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         б
NRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         г
HRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeGRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0WRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ╔
DRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereQRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ф
FRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeLRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
О
LRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
GRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0ORaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0URaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         У
%RaggedSegmentJoin/UnsortedSegmentJoinUnsortedSegmentJoin!RaggedFromTensor/Reshape:output:0PRaggedSegmentJoin/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator а
StaticRegexReplaceStaticRegexReplace.RaggedSegmentJoin/UnsortedSegmentJoin:output:0*#
_output_shapes
:         *
pattern \#\#*
rewrite С
StaticRegexReplace_1StaticRegexReplaceStaticRegexReplace:output:0*#
_output_shapes
:         *
pattern	^ +| +$*
rewrite S
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
value	B B г
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace_1:output:0StringSplit/Const:output:0*<
_output_shapes*
(:         :         :p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┼
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskл
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:         в
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ╝
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/SizeSizeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
: Т
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : л
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Size:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ╫
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: Ц
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: С
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: О
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :а
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: У
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: Ф
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: Ш
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: з
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         о
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ReshapeReshapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape/shape:output:0*
T0*#
_output_shapes
:         С
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 е
TStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincountDenseBincountWStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Reshape:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*

Tidx0*
T0	*#
_output_shapes
:         Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : п
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsum]StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/DenseBincount:output:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Ч
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R Л
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         д
StaticRegexFullMatchStaticRegexFullMatch"StringSplit/StringSplitV2:values:0*#
_output_shapes
:         *-
pattern" \[PAD\]|\[CLS\]|\[SEP\]|\[MASK\]\

LogicalNot
LogicalNotStaticRegexFullMatch:output:0*#
_output_shapes
:         b
RaggedMask/assert_equal/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Г
RaggedMask/CastCastLogicalNot:y:0^RaggedMask/assert_equal/NoOp*

DstT0	*

SrcT0
*#
_output_shapes
:         ╩
 RaggedMask/RaggedReduceSum/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
::э╧Ч
.RaggedMask/RaggedReduceSum/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Щ
0RaggedMask/RaggedReduceSum/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedMask/RaggedReduceSum/strided_sliceStridedSlice)RaggedMask/RaggedReduceSum/Shape:output:07RaggedMask/RaggedReduceSum/strided_slice/stack:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_1:output:09RaggedMask/RaggedReduceSum/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskБ
 RaggedMask/RaggedReduceSum/sub/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :д
RaggedMask/RaggedReduceSum/subSub1RaggedMask/RaggedReduceSum/strided_slice:output:0)RaggedMask/RaggedReduceSum/sub/y:output:0*
T0*
_output_shapes
: ░
GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:х
ARaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_sliceStridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_mask▓
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╜
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:я
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЖ
7RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/subSubJRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice:output:0LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         є
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/ShapeShapeMStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat:output:0^RaggedMask/assert_equal/NoOp*
T0	*
_output_shapes
:*
out_type0	:э╨╗
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ┤
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:▌
CRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2StridedSliceBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Shape:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0TRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskЬ
;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/yConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0	*
value	B	 Rї
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1SubLRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/strided_slice_2:output:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/startConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/deltaConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :└
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/CastCastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┬
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1CastHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: ╝
9RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/rangeRangeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast:y:0=RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub_1:z:0DRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ┴
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/CastCast;RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         └
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ShapeShapeBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧╖
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╣
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:°
HRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceIRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Shape:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╒
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackQRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:Я
FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastToCRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Cast:y:0URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         й
@RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: В
>RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaxMaxORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: а
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zeroConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : Б
BRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/zero:output:0GRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: о
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ░
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :ї
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         └
VRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         ─
RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Є
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCast[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         ╣
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  к
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
ERaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDimsBRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/range:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         м
KRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B :Э
IRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackTRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0FRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:ж
?RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/TileTileNRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0RRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ╙
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧─
[RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:е
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceVRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0dRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:╟
^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╛
LRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProd^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0gRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:┐
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╒
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧╞
]RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╚
_RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:╜
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceXRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0fRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0hRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskф
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:┤
SRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : М
NRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0\RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:л
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeHRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/Tile:output:0WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         ╔
WRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ╛
QRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapePRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0`RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         █
MRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereZRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         Ў
ORaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeURaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
╢
URaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ┴
PRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0XRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0^RaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         Х
-RaggedMask/RaggedReduceSum/UnsortedSegmentSumUnsortedSegmentSumRaggedMask/Cast:y:0YRaggedMask/RaggedReduceSum/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0"RaggedMask/RaggedReduceSum/sub:z:0*
Tindices0	*
T0	*#
_output_shapes
:         w
RaggedMask/Cumsum/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : в
RaggedMask/CumsumCumsum6RaggedMask/RaggedReduceSum/UnsortedSegmentSum:output:0RaggedMask/Cumsum/axis:output:0*
T0	*#
_output_shapes
:         Г
RaggedMask/concat/values_0Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0	*
valueB	R А
RaggedMask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
valueB :
         │
RaggedMask/concatConcatV2#RaggedMask/concat/values_0:output:0RaggedMask/Cumsum:out:0RaggedMask/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         з
(RaggedMask/RaggedMask/boolean_mask/ShapeShape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧Я
6RaggedMask/RaggedMask/boolean_mask/strided_slice/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:б
8RaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:ь
0RaggedMask/RaggedMask/boolean_mask/strided_sliceStridedSlice1RaggedMask/RaggedMask/boolean_mask/Shape:output:0?RaggedMask/RaggedMask/boolean_mask/strided_slice/stack:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:в
9RaggedMask/RaggedMask/boolean_mask/Prod/reduction_indicesConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: ╧
'RaggedMask/RaggedMask/boolean_mask/ProdProd9RaggedMask/RaggedMask/boolean_mask/strided_slice:output:0BRaggedMask/RaggedMask/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: й
*RaggedMask/RaggedMask/boolean_mask/Shape_1Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Ж
2RaggedMask/RaggedMask/boolean_mask/strided_slice_1StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_1:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskй
*RaggedMask/RaggedMask/boolean_mask/Shape_2Shape"StringSplit/StringSplitV2:values:0^RaggedMask/assert_equal/NoOp*
T0*
_output_shapes
::э╧б
8RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stackConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB: г
:RaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2Const^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:Д
2RaggedMask/RaggedMask/boolean_mask/strided_slice_2StridedSlice3RaggedMask/RaggedMask/boolean_mask/Shape_2:output:0ARaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_1:output:0CRaggedMask/RaggedMask/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskЪ
2RaggedMask/RaggedMask/boolean_mask/concat/values_1Pack0RaggedMask/RaggedMask/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:П
.RaggedMask/RaggedMask/boolean_mask/concat/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : ╙
)RaggedMask/RaggedMask/boolean_mask/concatConcatV2;RaggedMask/RaggedMask/boolean_mask/strided_slice_1:output:0;RaggedMask/RaggedMask/boolean_mask/concat/values_1:output:0;RaggedMask/RaggedMask/boolean_mask/strided_slice_2:output:07RaggedMask/RaggedMask/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:╗
*RaggedMask/RaggedMask/boolean_mask/ReshapeReshape"StringSplit/StringSplitV2:values:02RaggedMask/RaggedMask/boolean_mask/concat:output:0*
T0*#
_output_shapes
:         д
2RaggedMask/RaggedMask/boolean_mask/Reshape_1/shapeConst^RaggedMask/assert_equal/NoOp*
_output_shapes
:*
dtype0*
valueB:
         ▓
,RaggedMask/RaggedMask/boolean_mask/Reshape_1ReshapeLogicalNot:y:0;RaggedMask/RaggedMask/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         С
(RaggedMask/RaggedMask/boolean_mask/WhereWhere5RaggedMask/RaggedMask/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         м
*RaggedMask/RaggedMask/boolean_mask/SqueezeSqueeze0RaggedMask/RaggedMask/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
С
0RaggedMask/RaggedMask/boolean_mask/GatherV2/axisConst^RaggedMask/assert_equal/NoOp*
_output_shapes
: *
dtype0*
value	B : н
+RaggedMask/RaggedMask/boolean_mask/GatherV2GatherV23RaggedMask/RaggedMask/boolean_mask/Reshape:output:03RaggedMask/RaggedMask/boolean_mask/Squeeze:output:09RaggedMask/RaggedMask/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         q
RaggedSegmentJoin_1/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
::э╧q
'RaggedSegmentJoin_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedSegmentJoin_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedSegmentJoin_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedSegmentJoin_1/strided_sliceStridedSlice"RaggedSegmentJoin_1/Shape:output:00RaggedSegmentJoin_1/strided_slice/stack:output:02RaggedSegmentJoin_1/strided_slice/stack_1:output:02RaggedSegmentJoin_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
RaggedSegmentJoin_1/sub/yConst*
_output_shapes
: *
dtype0*
value	B :П
RaggedSegmentJoin_1/subSub*RaggedSegmentJoin_1/strided_slice:output:0"RaggedSegmentJoin_1/sub/y:output:0*
T0*
_output_shapes
: К
@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: М
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ц
:RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_sliceStridedSliceRaggedMask/concat:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_1:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskМ
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Ч
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:а
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1StridedSliceRaggedMask/concat:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskё
0RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/subSubCRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice:output:0ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_1:output:0*
T0	*#
_output_shapes
:         Ъ
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/ShapeShapeRaggedMask/concat:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Х
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
         О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: О
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:║
<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2StridedSlice;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Shape:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_1:output:0MRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskv
4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rр
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1SubERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/strided_slice_2:output:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1/y:output:0*
T0	*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/startConst*
_output_shapes
: *
dtype0*
value	B : z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▓
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/CastCastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: ┤
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1CastARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: а
2RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/rangeRange;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast:y:06RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub_1:z:0=RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         │
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/CastCast4RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/sub:z:0*

DstT0*

SrcT0	*#
_output_shapes
:         ▓
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ShapeShape;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0*
T0	*
_output_shapes
::э╧С
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:У
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╒
ARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_sliceStridedSliceBRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Shape:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_1:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╟
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shapePackJRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/strided_slice:output:0*
N*
T0*
_output_shapes
:К
?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastToBroadcastTo<RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Cast:y:0NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo/shape:output:0*
T0*#
_output_shapes
:         Г
9RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ConstConst*
_output_shapes
:*
dtype0*
valueB: э
7RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaxMaxHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Const:output:0*
T0*
_output_shapes
: z
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zeroConst*
_output_shapes
: *
dtype0*
value	B : ь
;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/MaximumMaximumARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/zero:output:0@RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Max:output:0*
T0*
_output_shapes
: И
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ConstConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1Const*
_output_shapes
: *
dtype0*
value	B :┘
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/RangeRangeORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Const_1:output:0*#
_output_shapes
:         Ъ
ORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         п
KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims
ExpandDimsHRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/BroadcastTo:output:0XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         ф
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/CastCastTRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/ExpandDims:output:0*

DstT0*

SrcT0*'
_output_shapes
:         д
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/LessLessORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Range:output:0IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Cast:y:0*
T0*0
_output_shapes
:                  Д
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :И
>RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims
ExpandDims;RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/range:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:         Ж
DRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :И
BRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiplesPackMRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples/0:output:0?RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Maximum:z:0*
N*
T0*
_output_shapes
:С
8RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/TileTileGRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/ExpandDims:output:0KRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile/multiples:output:0*
T0	*0
_output_shapes
:                  ┼
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ShapeShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧Ю
TRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:В
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_sliceStridedSliceORaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape:output:0]RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:б
WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: й
ERaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ProdProdWRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice:output:0`RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: ╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_1:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask╟
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2ShapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0*
T0	*
_output_shapes
::э╧а
VRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: в
XRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2StridedSliceQRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Shape_2:output:0_RaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_1:output:0aRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask╓
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1PackNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:О
LRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : щ
GRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concatConcatV2YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/values_1:output:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/strided_slice_2:output:0URaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:Ц
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/ReshapeReshapeARaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/Tile:output:0PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/concat:output:0*
T0	*#
_output_shapes
:         г
PRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         й
JRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1ReshapeIRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/SequenceMask/Less:z:0YRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:         ═
FRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/WhereWhereSRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape_1:output:0*'
_output_shapes
:         ш
HRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/SqueezeSqueezeNRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Where:index:0*
T0	*#
_output_shapes
:         *
squeeze_dims
Р
NRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : е
IRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2GatherV2QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Reshape:output:0QRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/Squeeze:output:0WRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         м
'RaggedSegmentJoin_1/UnsortedSegmentJoinUnsortedSegmentJoin4RaggedMask/RaggedMask/boolean_mask/GatherV2:output:0RRaggedSegmentJoin_1/RaggedSplitsToSegmentIds/Repeat/boolean_mask/GatherV2:output:0RaggedSegmentJoin_1/sub:z:0*
Tindices0	*#
_output_shapes
:         *
	separator {
IdentityIdentity0RaggedSegmentJoin_1/UnsortedSegmentJoin:output:0^NoOp*
T0*#
_output_shapes
:         T
NoOpNoOp^Assert/Assert ^None_Export/LookupTableExportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:                  : 2
Assert/AssertAssert/Assert2B
None_Export/LookupTableExportV2None_Export/LookupTableExportV2:,(
&
_user_specified_nametable_handle:[ W
0
_output_shapes
:                  
#
_user_specified_name	tokenized
ў
║
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5812╨
╦raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_3_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
З
м
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5649в
Эraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All
╥	
╪
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_4515[
Wraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all
>
:raggedconcat_assert_equal_3_assert_assertguard_placeholder	@
<raggedconcat_assert_equal_3_assert_assertguard_placeholder_1	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
y
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityWraggedconcat_assert_equal_3_assert_assertguard_identity_raggedconcat_assert_equal_3_all4^RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: и
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_3/All
Ў
a
__inference_get_vocab_size_3857-
shape_readvariableop_resource:
░ъ
identityИp
Shape/ReadVariableOpReadVariableOpshape_readvariableop_resource*
_output_shapes

:░ъ*
dtype0Q
ShapeConst*
_output_shapes
:*
dtype0*
valueB:░ъ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskM
IdentityIdentitystrided_slice:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :( $
"
_user_specified_name
resource
Б
▄
__inference__initializer_6114*
&text_file_id_table_init_asset_filepathF
Btext_file_id_table_init_initializetablefromtextfilev2_table_handle
identityИв5text_file_id_table_init/InitializeTableFromTextFileV2О
5text_file_id_table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2Btext_file_id_table_init_initializetablefromtextfilev2_table_handle&text_file_id_table_init_asset_filepath*
_output_shapes
 *
	key_index■        *
value_index         G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: Z
NoOpNoOp6^text_file_id_table_init/InitializeTableFromTextFileV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2n
5text_file_id_table_init/InitializeTableFromTextFileV25text_file_id_table_init/InitializeTableFromTextFileV2:,(
&
_user_specified_nametable_handle: 

_output_shapes
: 
╖
П
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_4090k
graggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all
h
draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1	f
braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = ▓
ERaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = К
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_allNRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice_1NRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_1_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_1_assert_equal_1_all?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_1/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_1/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_1/assert_equal_1/All
ў
║
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5685╨
╦raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
x
traggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_placeholder	w
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
│
mRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╦raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_raggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_alln^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ь
sRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "є
sraggedfromrowsplits_2_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         :)%
#
_output_shapes
:         :Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
з"
▀
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5573╬
╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_all
и
гraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub	w
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1
ИвoRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertБ
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  т
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:є
vRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = ╗
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertAssert╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_0:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_1:output:0RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert/data_2:output:0гraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_sub*
T
2	*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentity╔raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_assert_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_allp^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: М
sRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1IdentityzRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity:output:0n^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ¤
mRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/NoOpNoOpp^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert*
_output_shapes
 "є
sraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_non_negative_assert_less_equal_assert_assertguard_identity_1|RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
: :         2т
oRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/AssertoRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Assert:xt
#
_output_shapes
:         
M
_user_specified_name53RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:Т Н

_output_shapes
: 
s
_user_specified_name[YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All
З

о
__inference_lookup_5393
	token_ids	
token_ids_1	4
$raggedgather_readvariableop_resource:
░ъ
identity

identity_1	ИвRaggedGather/ReadVariableOp~
RaggedGather/ReadVariableOpReadVariableOp$raggedgather_readvariableop_resource*
_output_shapes

:░ъ*
dtype0\
RaggedGather/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╟
RaggedGather/GatherV2GatherV2#RaggedGather/ReadVariableOp:value:0	token_ids#RaggedGather/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:         i
IdentityIdentityRaggedGather/GatherV2:output:0^NoOp*
T0*#
_output_shapes
:         X

Identity_1Identitytoken_ids_1^NoOp*
T0	*#
_output_shapes
:         @
NoOpNoOp^RaggedGather/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         :         : 2:
RaggedGather/ReadVariableOpRaggedGather/ReadVariableOp:($
"
_user_specified_name
resource:NJ
#
_output_shapes
:         
#
_user_specified_name	token_ids:N J
#
_output_shapes
:         
#
_user_specified_name	token_ids
З
м
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4016в
Эraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_all
a
]raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder	c
_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_placeholder_1	`
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1
Ь
VRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 А
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityЭraggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_allW^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ю
\RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1IdentitycRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "┼
\raggedfromrowsplits_1_rowpartitionfromrowsplits_assert_equal_1_assert_assertguard_identity_1eRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :z v

_output_shapes
: 
\
_user_specified_nameDBRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All
╖
П
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_5850k
graggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all
h
draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1	f
braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice	F
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1
Ив>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert╟
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensor▒
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:┤
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = ▓
ERaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = К
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/AssertAssertgraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_allNRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_0:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_1:output:0NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_2:output:0draggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice_1NRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert/data_4:output:0braggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_strided_slice*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentitygraggedfromrowsplits_3_assert_equal_1_assert_assertguard_assert_raggedfromrowsplits_3_assert_equal_1_all?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ∙
BRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity:output:0=^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Ы
<RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/NoOpNoOp?^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert*
_output_shapes
 "С
Braggedfromrowsplits_3_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2А
>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert>RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/Assert:[W

_output_shapes
: 
=
_user_specified_name%#RaggedFromRowSplits_3/strided_slice:]Y

_output_shapes
: 
?
_user_specified_name'%RaggedFromRowSplits_3/strided_slice_1:` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_3/assert_equal_1/All
№
<
__inference_get_vocab_path_5367
unknown
identity>
IdentityIdentityunknown*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
Ц
c
__inference_lookup_5383
	token_ids	
gather_resource:
░ъ
identityИвGatherГ
GatherResourceGathergather_resource	token_ids*
Tindices0	*0
_output_shapes
:                  *
dtype0g
IdentityIdentityGather:output:0^NoOp*
T0*0
_output_shapes
:                  +
NoOpNoOp^Gather*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:                  : 2
GatherGather:($
"
_user_specified_name
resource:[ W
0
_output_shapes
:                  
#
_user_specified_name	token_ids
ь╟
┤
__inference_tokenize_4584
stringsm
iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tableu
qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value	

fill_value	
fill_1_value	
identity	

identity_1	Ив.RaggedConcat/assert_equal_1/Assert/AssertGuardв.RaggedConcat/assert_equal_3/Assert/AssertGuardвORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardвcWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2вaWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2вVWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsJ
ShapeShapestrings*
T0*
_output_shapes
::э╧P
CastCastShape:output:0*

DstT0	*

SrcT0*
_output_shapes
:`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         a
ReshapeReshapestringsReshape/shape:output:0*
T0*#
_output_shapes
:         n
)RegexSplitWithOffsets/delim_regex_patternConst*
_output_shapes
: *
dtype0*
valueB
 B(\s)q
.RegexSplitWithOffsets/keep_delim_regex_patternConst*
_output_shapes
: *
dtype0*
value
B B() 
RegexSplitWithOffsetsRegexSplitWithOffsetsReshape:output:02RegexSplitWithOffsets/delim_regex_pattern:output:07RegexSplitWithOffsets/keep_delim_regex_pattern:output:0*P
_output_shapes>
<:         :         :         :         А
>RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :Я
?RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧Ж
hRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Я
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Л
ARaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Э
;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0JRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_masku
3RaggedFromRowSplits/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ё
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/EqualEqualDRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Г
ARaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╫
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/rangeRangeQRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0JRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0QRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: Ї
@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/AllAllFRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╠
IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╖
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╨
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = ╚
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = ъ
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfIRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0DRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *m
else_branch^R\
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3904*
output_shapes
: *l
then_branch]R[
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3903З
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityXRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskН
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЎ
1RaggedFromRowSplits/RowPartitionFromRowSplits/subSubFRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1:output:0FRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Й
GRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R б
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualPRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/Const:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         г
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╜
WRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: ы
`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╬
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:▌
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = у
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIf`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0P^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Д
else_branchuRs
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_3940*
output_shapes
: *Г
then_branchtRr
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_3939╡
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityoRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ё
@RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0Y^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityp^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityZ^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         Е
RaggedFromRowSplits/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
:*
out_type0	:э╨q
'RaggedFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedFromRowSplits/strided_sliceStridedSlice"RaggedFromRowSplits/Shape:output:00RaggedFromRowSplits/strided_slice/stack:output:02RaggedFromRowSplits/strided_slice/stack_1:output:02RaggedFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask|
)RaggedFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         u
+RaggedFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
#RaggedFromRowSplits/strided_slice_1StridedSliceIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:02RaggedFromRowSplits/strided_slice_1/stack:output:04RaggedFromRowSplits/strided_slice_1/stack_1:output:04RaggedFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskм
(RaggedFromRowSplits/assert_equal_1/EqualEqual,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0*
T0	*
_output_shapes
: i
'RaggedFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : p
.RaggedFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.RaggedFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :я
(RaggedFromRowSplits/assert_equal_1/rangeRange7RaggedFromRowSplits/assert_equal_1/range/start:output:00RaggedFromRowSplits/assert_equal_1/Rank:output:07RaggedFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ж
&RaggedFromRowSplits/assert_equal_1/AllAll,RaggedFromRowSplits/assert_equal_1/Equal:z:01RaggedFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ▒
/RaggedFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЭ
1RaggedFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Ю
1RaggedFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = Ь
1RaggedFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = з
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardIf/RaggedFromRowSplits/assert_equal_1/All:output:0/RaggedFromRowSplits/assert_equal_1/All:output:0,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0g^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *S
else_branchDRB
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3977*
output_shapes
: *R
then_branchCRA
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3976╙
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: o
-RaggedFromRowSplits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :К
.RaggedFromRowSplits/assert_rank_at_least/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
::э╧u
WRaggedFromRowSplits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 О
HRaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 с
&RaggedFromRowSplits/control_dependencyIdentityIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:0?^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityI^RaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         В
@RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :б
ARaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_1/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = ░
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:06^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4017*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4016Л
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_1/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4053*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4052╣
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ∙
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         О
RaggedFromRowSplits_1/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_1/strided_sliceStridedSlice$RaggedFromRowSplits_1/Shape:output:02RaggedFromRowSplits_1/strided_slice/stack:output:04RaggedFromRowSplits_1/strided_slice/stack_1:output:04RaggedFromRowSplits_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_1/strided_slice_1StridedSliceKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_1/strided_slice_1/stack:output:06RaggedFromRowSplits_1/strided_slice_1/stack_1:output:06RaggedFromRowSplits_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_1/assert_equal_1/EqualEqual.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_1/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_1/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_1/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_1/assert_equal_1/rangeRange9RaggedFromRowSplits_1/assert_equal_1/range/start:output:02RaggedFromRowSplits_1/assert_equal_1/Rank:output:09RaggedFromRowSplits_1/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_1/assert_equal_1/AllAll.RaggedFromRowSplits_1/assert_equal_1/Equal:z:03RaggedFromRowSplits_1/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_1/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = а
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = ╖
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_1/assert_equal_1/All:output:01RaggedFromRowSplits_1/assert_equal_1/All:output:0.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0i^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_4090*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_4089╫
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_1/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :У
0RaggedFromRowSplits_1/assert_rank_at_least/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_1/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 щ
(RaggedFromRowSplits_1/control_dependencyIdentityKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         В
@RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :б
ARaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_2/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = ▓
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4130*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4129Л
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_2/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4166*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4165╣
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ∙
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         М
RaggedFromRowSplits_2/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_2/strided_sliceStridedSlice$RaggedFromRowSplits_2/Shape:output:02RaggedFromRowSplits_2/strided_slice/stack:output:04RaggedFromRowSplits_2/strided_slice/stack_1:output:04RaggedFromRowSplits_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_2/strided_slice_1StridedSliceKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_2/strided_slice_1/stack:output:06RaggedFromRowSplits_2/strided_slice_1/stack_1:output:06RaggedFromRowSplits_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_2/assert_equal_1/EqualEqual.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_2/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_2/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_2/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_2/assert_equal_1/rangeRange9RaggedFromRowSplits_2/assert_equal_1/range/start:output:02RaggedFromRowSplits_2/assert_equal_1/Rank:output:09RaggedFromRowSplits_2/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_2/assert_equal_1/AllAll.RaggedFromRowSplits_2/assert_equal_1/Equal:z:03RaggedFromRowSplits_2/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_2/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = а
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = ╖
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_2/assert_equal_1/All:output:01RaggedFromRowSplits_2/assert_equal_1/All:output:0.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0i^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_4203*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_4202╫
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_2/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :С
0RaggedFromRowSplits_2/assert_rank_at_least/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_2/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 щ
(RaggedFromRowSplits_2/control_dependencyIdentityKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         х
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsWordpieceTokenizeWithOffsetsRegexSplitWithOffsets:tokens:0iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_table*P
_output_shapes>
<:         :         :         :         *
max_bytes_per_wordd*)
output_row_partition_type
row_splits*
suffix_indicator##*
unknown_token[UNK]*
use_unknown_token(Л
QWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucketStringToHashBucketFastfWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0*#
_output_shapes
:         *
num_buckets╙
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tablefWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_valueW^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*	
Tin0*

Tout0	*#
_output_shapes
:         ▀
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tabled^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2*
_output_shapes
: ╞
IWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/AddAddV2ZWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucket:output:0hWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*#
_output_shapes
:         щ
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqualNotEquallWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*#
_output_shapes
:         Щ
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2SelectV2RWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqual:z:0lWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0MWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/Add:z:0*
T0	*#
_output_shapes
:         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
GatherV2GatherV2kWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_row_lengths:0/RaggedFromRowSplits/control_dependency:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         В
@RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :Р
ARaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/ShapeShapeGatherV2:output:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_sliceStridedSliceGatherV2:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_3/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = ▓
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4257*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4256Л
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_3/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4293*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4292╣
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: █
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependencyIdentityGatherV2:output:0[^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:         └
RaggedFromRowSplits_3/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_3/strided_sliceStridedSlice$RaggedFromRowSplits_3/Shape:output:02RaggedFromRowSplits_3/strided_slice/stack:output:04RaggedFromRowSplits_3/strided_slice/stack_1:output:04RaggedFromRowSplits_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_3/strided_slice_1StridedSliceKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_3/strided_slice_1/stack:output:06RaggedFromRowSplits_3/strided_slice_1/stack_1:output:06RaggedFromRowSplits_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_3/assert_equal_1/EqualEqual.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_3/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_3/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_3/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_3/assert_equal_1/rangeRange9RaggedFromRowSplits_3/assert_equal_1/range/start:output:02RaggedFromRowSplits_3/assert_equal_1/Rank:output:09RaggedFromRowSplits_3/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_3/assert_equal_1/AllAll.RaggedFromRowSplits_3/assert_equal_1/Equal:z:03RaggedFromRowSplits_3/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_3/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = а
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = ╖
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_3/assert_equal_1/All:output:01RaggedFromRowSplits_3/assert_equal_1/All:output:0.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0i^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_4330*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_4329╫
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_3/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :┼
0RaggedFromRowSplits_3/assert_rank_at_least/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_3/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
(RaggedFromRowSplits_3/control_dependencyIdentityKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:         Ц
RaggedBoundingBox/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨╛
RaggedBoundingBox/Shape_1ShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨o
%RaggedBoundingBox/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedBoundingBox/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedBoundingBox/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedBoundingBox/strided_sliceStridedSlice RaggedBoundingBox/Shape:output:0.RaggedBoundingBox/strided_slice/stack:output:00RaggedBoundingBox/strided_slice/stack_1:output:00RaggedBoundingBox/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskY
RaggedBoundingBox/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЙ
RaggedBoundingBox/subSub(RaggedBoundingBox/strided_slice:output:0 RaggedBoundingBox/sub/y:output:0*
T0	*
_output_shapes
: q
'RaggedBoundingBox/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)RaggedBoundingBox/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)RaggedBoundingBox/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
!RaggedBoundingBox/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_1/stack:output:02RaggedBoundingBox/strided_slice_1/stack_1:output:02RaggedBoundingBox/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskq
'RaggedBoundingBox/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
)RaggedBoundingBox/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         s
)RaggedBoundingBox/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
!RaggedBoundingBox/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_2/stack:output:02RaggedBoundingBox/strided_slice_2/stack_1:output:02RaggedBoundingBox/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskд
RaggedBoundingBox/sub_1Sub*RaggedBoundingBox/strided_slice_1:output:0*RaggedBoundingBox/strided_slice_2:output:0*
T0	*#
_output_shapes
:         a
RaggedBoundingBox/ConstConst*
_output_shapes
:*
dtype0*
valueB: |
RaggedBoundingBox/MaxMaxRaggedBoundingBox/sub_1:z:0 RaggedBoundingBox/Const:output:0*
T0	*
_output_shapes
: ]
RaggedBoundingBox/Maximum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Л
RaggedBoundingBox/MaximumMaximumRaggedBoundingBox/Max:output:0$RaggedBoundingBox/Maximum/y:output:0*
T0	*
_output_shapes
: q
'RaggedBoundingBox/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)RaggedBoundingBox/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)RaggedBoundingBox/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
!RaggedBoundingBox/strided_slice_3StridedSlice"RaggedBoundingBox/Shape_1:output:00RaggedBoundingBox/strided_slice_3/stack:output:02RaggedBoundingBox/strided_slice_3/stack_1:output:02RaggedBoundingBox/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskЗ
RaggedBoundingBox/stackPackRaggedBoundingBox/sub:z:0RaggedBoundingBox/Maximum:z:0*
N*
T0	*
_output_shapes
:_
RaggedBoundingBox/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╚
RaggedBoundingBox/concatConcatV2 RaggedBoundingBox/stack:output:0*RaggedBoundingBox/strided_slice_3:output:0&RaggedBoundingBox/concat/axis:output:0*
N*
T0	*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
strided_sliceStridedSlice!RaggedBoundingBox/concat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskM
Fill/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 Rm
	Fill/dimsPackstrided_slice:output:0Fill/dims/1:output:0*
N*
T0	*
_output_shapes
:p
FillFillFill/dims:output:0
fill_value*
T0	*'
_output_shapes
:         *

index_type0	O
Fill_1/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 Rq
Fill_1/dimsPackstrided_slice:output:0Fill_1/dims/1:output:0*
N*
T0	*
_output_shapes
:v
Fill_1FillFill_1/dims:output:0fill_1_value*
T0	*'
_output_shapes
:         *

index_type0	~
#RaggedConcat/RaggedFromTensor/ShapeShapeFill:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨{
1RaggedConcat/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3RaggedConcat/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3RaggedConcat/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+RaggedConcat/RaggedFromTensor/strided_sliceStridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0:RaggedConcat/RaggedFromTensor/strided_slice/stack:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_1:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask}
3RaggedConcat/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_1StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_1/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask}
3RaggedConcat/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_2StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_2/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╣
!RaggedConcat/RaggedFromTensor/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_1:output:06RaggedConcat/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: }
3RaggedConcat/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
-RaggedConcat/RaggedFromTensor/strided_slice_3StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_3/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskК
-RaggedConcat/RaggedFromTensor/concat/values_0Pack%RaggedConcat/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:k
)RaggedConcat/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : В
$RaggedConcat/RaggedFromTensor/concatConcatV26RaggedConcat/RaggedFromTensor/concat/values_0:output:06RaggedConcat/RaggedFromTensor/strided_slice_3:output:02RaggedConcat/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:к
%RaggedConcat/RaggedFromTensor/ReshapeReshapeFill:output:0-RaggedConcat/RaggedFromTensor/concat:output:0*
Tshape0	*
T0	*#
_output_shapes
:         }
3RaggedConcat/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_4StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_4/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#RaggedConcat/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R║
>RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape.RaggedConcat/RaggedFromTensor/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Ц
LRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
FRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSliceGRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0URaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rй
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV26RaggedConcat/RaggedFromTensor/strided_slice_4:output:0hRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R▄
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangenRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0nRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         к
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMulhRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0,RaggedConcat/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:         В
%RaggedConcat/RaggedFromTensor_1/ShapeShapeFill_1:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨}
3RaggedConcat/RaggedFromTensor_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
-RaggedConcat/RaggedFromTensor_1/strided_sliceStridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0<RaggedConcat/RaggedFromTensor_1/strided_slice/stack:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
5RaggedConcat/RaggedFromTensor_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_1StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
5RaggedConcat/RaggedFromTensor_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_2StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask┐
#RaggedConcat/RaggedFromTensor_1/mulMul8RaggedConcat/RaggedFromTensor_1/strided_slice_1:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_2:output:0*
T0	*
_output_shapes
: 
5RaggedConcat/RaggedFromTensor_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
/RaggedConcat/RaggedFromTensor_1/strided_slice_3StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskО
/RaggedConcat/RaggedFromTensor_1/concat/values_0Pack'RaggedConcat/RaggedFromTensor_1/mul:z:0*
N*
T0	*
_output_shapes
:m
+RaggedConcat/RaggedFromTensor_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : К
&RaggedConcat/RaggedFromTensor_1/concatConcatV28RaggedConcat/RaggedFromTensor_1/concat/values_0:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_3:output:04RaggedConcat/RaggedFromTensor_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:░
'RaggedConcat/RaggedFromTensor_1/ReshapeReshapeFill_1:output:0/RaggedConcat/RaggedFromTensor_1/concat:output:0*
Tshape0	*
T0	*#
_output_shapes
:         
5RaggedConcat/RaggedFromTensor_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_4StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%RaggedConcat/RaggedFromTensor_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R╛
@RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/ShapeShape0RaggedConcat/RaggedFromTensor_1/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Ш
NRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
HRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_sliceStridedSliceIRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/Shape:output:0WRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskг
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rп
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV28RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0jRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: й
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R й
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 Rф
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangepRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0cRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0pRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         ░
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMuljRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0.RaggedConcat/RaggedFromTensor_1/Const:output:0*
T0	*#
_output_shapes
:         Я
 RaggedConcat/RaggedNRows_1/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨x
.RaggedConcat/RaggedNRows_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0RaggedConcat/RaggedNRows_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0RaggedConcat/RaggedNRows_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedConcat/RaggedNRows_1/strided_sliceStridedSlice)RaggedConcat/RaggedNRows_1/Shape:output:07RaggedConcat/RaggedNRows_1/strided_slice/stack:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_1:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 RaggedConcat/RaggedNRows_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rд
RaggedConcat/RaggedNRows_1/subSub1RaggedConcat/RaggedNRows_1/strided_slice:output:0)RaggedConcat/RaggedNRows_1/sub/y:output:0*
T0	*
_output_shapes
: з
!RaggedConcat/assert_equal_1/EqualEqual6RaggedConcat/RaggedFromTensor/strided_slice_4:output:0"RaggedConcat/RaggedNRows_1/sub:z:0*
T0	*
_output_shapes
: b
 RaggedConcat/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╙
!RaggedConcat/assert_equal_1/rangeRange0RaggedConcat/assert_equal_1/range/start:output:0)RaggedConcat/assert_equal_1/Rank:output:00RaggedConcat/assert_equal_1/range/delta:output:0*
_output_shapes
: С
RaggedConcat/assert_equal_1/AllAll%RaggedConcat/assert_equal_1/Equal:z:0*RaggedConcat/assert_equal_1/range:output:0*
_output_shapes
: л
(RaggedConcat/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 1 (=y) have incompatible shapes.Ц
*RaggedConcat/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:б
*RaggedConcat/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = Т
*RaggedConcat/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*8
value/B- B'y (RaggedConcat/RaggedNRows_1/sub:0) = ╫
.RaggedConcat/assert_equal_1/Assert/AssertGuardIf(RaggedConcat/assert_equal_1/All:output:0(RaggedConcat/assert_equal_1/All:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:0"RaggedConcat/RaggedNRows_1/sub:z:08^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_4486*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_4485┼
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ╜
!RaggedConcat/assert_equal_3/EqualEqual6RaggedConcat/RaggedFromTensor/strided_slice_4:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0*
T0	*
_output_shapes
: b
 RaggedConcat/assert_equal_3/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╙
!RaggedConcat/assert_equal_3/rangeRange0RaggedConcat/assert_equal_3/range/start:output:0)RaggedConcat/assert_equal_3/Rank:output:00RaggedConcat/assert_equal_3/range/delta:output:0*
_output_shapes
: С
RaggedConcat/assert_equal_3/AllAll%RaggedConcat/assert_equal_3/Equal:z:0*RaggedConcat/assert_equal_3/range:output:0*
_output_shapes
: л
(RaggedConcat/assert_equal_3/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 2 (=y) have incompatible shapes.Ц
*RaggedConcat/assert_equal_3/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:б
*RaggedConcat/assert_equal_3/Assert/Const_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = г
*RaggedConcat/assert_equal_3/Assert/Const_3Const*
_output_shapes
: *
dtype0*I
value@B> B8y (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = ф
.RaggedConcat/assert_equal_3/Assert/AssertGuardIf(RaggedConcat/assert_equal_3/All:output:0(RaggedConcat/assert_equal_3/All:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0/^RaggedConcat/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_4516*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_4515┼
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_3/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ╬
RaggedConcat/concat/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ┤
RaggedConcat/concatConcatV2.RaggedConcat/RaggedFromTensor/Reshape:output:0WWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:00RaggedConcat/RaggedFromTensor_1/Reshape:output:0!RaggedConcat/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ч
 RaggedConcat/strided_slice/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         р
"RaggedConcat/strided_slice/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: р
"RaggedConcat/strided_slice/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╪
RaggedConcat/strided_sliceStridedSliceaRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0)RaggedConcat/strided_slice/stack:output:0+RaggedConcat/strided_slice/stack_1:output:0+RaggedConcat/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskр
"RaggedConcat/strided_slice_1/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
$RaggedConcat/strided_slice_1/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_1/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╡
RaggedConcat/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_1/stack:output:0-RaggedConcat/strided_slice_1/stack_1:output:0-RaggedConcat/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskУ
RaggedConcat/addAddV2%RaggedConcat/strided_slice_1:output:0#RaggedConcat/strided_slice:output:0*
T0	*#
_output_shapes
:         щ
"RaggedConcat/strided_slice_2/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         т
$RaggedConcat/strided_slice_2/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_2/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:░
RaggedConcat/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_2/stack:output:0-RaggedConcat/strided_slice_2/stack_1:output:0-RaggedConcat/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskИ
RaggedConcat/add_1AddV2#RaggedConcat/strided_slice:output:0%RaggedConcat/strided_slice_2:output:0*
T0	*
_output_shapes
: р
"RaggedConcat/strided_slice_3/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
$RaggedConcat/strided_slice_3/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_3/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:ч
RaggedConcat/strided_slice_3StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_3/stack:output:0-RaggedConcat/strided_slice_3/stack_1:output:0-RaggedConcat/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskИ
RaggedConcat/add_2AddV2%RaggedConcat/strided_slice_3:output:0RaggedConcat/add_1:z:0*
T0	*#
_output_shapes
:         щ
"RaggedConcat/strided_slice_4/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         т
$RaggedConcat/strided_slice_4/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_4/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
RaggedConcat/strided_slice_4StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_4/stack:output:0-RaggedConcat/strided_slice_4/stack_1:output:0-RaggedConcat/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask{
RaggedConcat/add_3AddV2RaggedConcat/add_1:z:0%RaggedConcat/strided_slice_4:output:0*
T0	*
_output_shapes
: ╨
RaggedConcat/concat_1/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : О
RaggedConcat/concat_1ConcatV2aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0RaggedConcat/add:z:0RaggedConcat/add_2:z:0#RaggedConcat/concat_1/axis:output:0*
N*
T0	*#
_output_shapes
:         ╚
RaggedConcat/mul/yConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RН
RaggedConcat/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_4:output:0RaggedConcat/mul/y:output:0*
T0	*
_output_shapes
: ╬
RaggedConcat/range/startConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ╬
RaggedConcat/range/deltaConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :r
RaggedConcat/range/CastCast!RaggedConcat/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: t
RaggedConcat/range/Cast_1Cast!RaggedConcat/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ю
RaggedConcat/rangeRangeRaggedConcat/range/Cast:y:0RaggedConcat/mul:z:0RaggedConcat/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ▀
RaggedConcat/Reshape/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       У
RaggedConcat/ReshapeReshapeRaggedConcat/range:output:0#RaggedConcat/Reshape/shape:output:0*
T0	*'
_output_shapes
:         р
RaggedConcat/transpose/permConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       Ъ
RaggedConcat/transpose	TransposeRaggedConcat/Reshape:output:0$RaggedConcat/transpose/perm:output:0*
T0	*'
_output_shapes
:         у
RaggedConcat/Reshape_1/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         Т
RaggedConcat/Reshape_1ReshapeRaggedConcat/transpose:y:0%RaggedConcat/Reshape_1/shape:output:0*
T0	*#
_output_shapes
:         Ь
&RaggedConcat/RaggedGather/RaggedGatherRaggedGatherRaggedConcat/concat_1:output:0RaggedConcat/concat:output:0RaggedConcat/Reshape_1:output:0*
OUTPUT_RAGGED_RANK*
PARAMS_RAGGED_RANK*
Tindices0	*
Tvalues0	*2
_output_shapes 
:         :         р
"RaggedConcat/strided_slice_5/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_5/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_5/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╙
RaggedConcat/strided_slice_5StridedSlice=RaggedConcat/RaggedGather/RaggedGather:output_nested_splits:0+RaggedConcat/strided_slice_5/stack:output:0-RaggedConcat/strided_slice_5/stack_1:output:0-RaggedConcat/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_maskЗ
IdentityIdentity<RaggedConcat/RaggedGather/RaggedGather:output_dense_values:0^NoOp*
T0	*#
_output_shapes
:         r

Identity_1Identity%RaggedConcat/strided_slice_5:output:0^NoOp*
T0	*#
_output_shapes
:         Е
NoOpNoOp/^RaggedConcat/assert_equal_1/Assert/AssertGuard/^RaggedConcat/assert_equal_3/Assert/AssertGuardP^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardg^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard6^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardd^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2b^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2W^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : : : 2`
.RaggedConcat/assert_equal_1/Assert/AssertGuard.RaggedConcat/assert_equal_1/Assert/AssertGuard2`
.RaggedConcat/assert_equal_3/Assert/AssertGuard.RaggedConcat/assert_equal_3/Assert/AssertGuard2в
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╨
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2n
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard2╩
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV22╞
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV22░
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsVWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :2.
,
_user_specified_namevocab_lookup_table:L H
#
_output_shapes
:         
!
_user_specified_name	strings
╘
ф
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_6036Y
Uraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all
g
craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4	i
eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4	=
9raggedconcat_assert_equal_3_assert_assertguard_identity_1
Ив5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert┐
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 2 (=y) have incompatible shapes.и
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:│
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = ╡
<RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*I
value@B> B8y (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = ═
5RaggedConcat/assert_equal_3/Assert/AssertGuard/AssertAssertUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_allERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_0:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_1:output:0ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_2:output:0craggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_strided_slice_4ERaggedConcat/assert_equal_3/Assert/AssertGuard/Assert/data_4:output:0eraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_raggedfromtensor_1_strided_slice_4*
T

2		*&
 _has_manual_control_dependencies(*
_output_shapes
 є
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentityUraggedconcat_assert_equal_3_assert_assertguard_assert_raggedconcat_assert_equal_3_all6^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: ▐
9RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1Identity@RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity:output:04^RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: Й
3RaggedConcat/assert_equal_3/Assert/AssertGuard/NoOpNoOp6^RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert*
_output_shapes
 "
9raggedconcat_assert_equal_3_assert_assertguard_identity_1BRaggedConcat/assert_equal_3/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2n
5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert5RaggedConcat/assert_equal_3/Assert/AssertGuard/Assert:gc

_output_shapes
: 
I
_user_specified_name1/RaggedConcat/RaggedFromTensor_1/strided_slice_4:ea

_output_shapes
: 
G
_user_specified_name/-RaggedConcat/RaggedFromTensor/strided_slice_4:W S

_output_shapes
: 
9
_user_specified_name!RaggedConcat/assert_equal_3/All
ф

О
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_4089m
iraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all
G
Craggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder	I
Eraggedfromrowsplits_1_assert_equal_1_assert_assertguard_placeholder_1	F
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1
В
<RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOpNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Ч
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentityiraggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_raggedfromrowsplits_1_assert_equal_1_all=^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: ║
BRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1IdentityIRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: "С
Braggedfromrowsplits_1_assert_equal_1_assert_assertguard_identity_1KRaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : :

_output_shapes
: :

_output_shapes
: :` \

_output_shapes
: 
B
_user_specified_name*(RaggedFromRowSplits_1/assert_equal_1/All
ь╟
┤
__inference_tokenize_6104
stringsm
iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tableu
qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value	

fill_value	
fill_1_value	
identity	

identity_1	Ив.RaggedConcat/assert_equal_1/Assert/AssertGuardв.RaggedConcat/assert_equal_3/Assert/AssertGuardвORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardвQRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardвhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardв7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardвcWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2вaWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2вVWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsJ
ShapeShapestrings*
T0*
_output_shapes
::э╧P
CastCastShape:output:0*

DstT0	*

SrcT0*
_output_shapes
:`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         a
ReshapeReshapestringsReshape/shape:output:0*
T0*#
_output_shapes
:         n
)RegexSplitWithOffsets/delim_regex_patternConst*
_output_shapes
: *
dtype0*
valueB
 B(\s)q
.RegexSplitWithOffsets/keep_delim_regex_patternConst*
_output_shapes
: *
dtype0*
value
B B() 
RegexSplitWithOffsetsRegexSplitWithOffsetsReshape:output:02RegexSplitWithOffsets/delim_regex_pattern:output:07RegexSplitWithOffsets/keep_delim_regex_pattern:output:0*P
_output_shapes>
<:         :         :         :         А
>RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :Я
?RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧Ж
hRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Я
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Л
ARaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Э
;RaggedFromRowSplits/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0JRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_1:output:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_masku
3RaggedFromRowSplits/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ё
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/EqualEqualDRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Г
ARaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : К
HRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╫
BRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/rangeRangeQRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0JRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0QRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: Ї
@RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/AllAllFRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╠
IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╖
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╨
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*U
valueLBJ BDx (RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = ╚
KRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*M
valueDBB B<y (RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = ъ
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfIRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0IRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/All:output:0DRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:output:0<RaggedFromRowSplits/RowPartitionFromRowSplits/Const:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *m
else_branch^R\
ZRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5424*
output_shapes
: *l
then_branch]R[
YRaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5423З
XRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityXRaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: Н
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskН
CRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         П
ERaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:м
=RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0NRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskЎ
1RaggedFromRowSplits/RowPartitionFromRowSplits/subSubFRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_1:output:0FRaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Й
GRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R б
]RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualPRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/Const:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         г
YRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ╜
WRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllaRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: ы
`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╬
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:▌
bRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = у
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIf`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0`RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:05RaggedFromRowSplits/RowPartitionFromRowSplits/sub:z:0P^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Д
else_branchuRs
qRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5460*
output_shapes
: *Г
then_branchtRr
pRaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5459╡
oRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityoRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ё
@RaggedFromRowSplits/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0Y^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityp^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityZ^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         Е
RaggedFromRowSplits/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
:*
out_type0	:э╨q
'RaggedFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)RaggedFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)RaggedFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╡
!RaggedFromRowSplits/strided_sliceStridedSlice"RaggedFromRowSplits/Shape:output:00RaggedFromRowSplits/strided_slice/stack:output:02RaggedFromRowSplits/strided_slice/stack_1:output:02RaggedFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask|
)RaggedFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         u
+RaggedFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
#RaggedFromRowSplits/strided_slice_1StridedSliceIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:02RaggedFromRowSplits/strided_slice_1/stack:output:04RaggedFromRowSplits/strided_slice_1/stack_1:output:04RaggedFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskм
(RaggedFromRowSplits/assert_equal_1/EqualEqual,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0*
T0	*
_output_shapes
: i
'RaggedFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : p
.RaggedFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : p
.RaggedFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :я
(RaggedFromRowSplits/assert_equal_1/rangeRange7RaggedFromRowSplits/assert_equal_1/range/start:output:00RaggedFromRowSplits/assert_equal_1/Rank:output:07RaggedFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ж
&RaggedFromRowSplits/assert_equal_1/AllAll,RaggedFromRowSplits/assert_equal_1/Equal:z:01RaggedFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ▒
/RaggedFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЭ
1RaggedFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Ю
1RaggedFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (RaggedFromRowSplits/strided_slice_1:0) = Ь
1RaggedFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*;
value2B0 B*y (RaggedFromRowSplits/strided_slice:0) = з
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardIf/RaggedFromRowSplits/assert_equal_1/All:output:0/RaggedFromRowSplits/assert_equal_1/All:output:0,RaggedFromRowSplits/strided_slice_1:output:0*RaggedFromRowSplits/strided_slice:output:0g^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *S
else_branchDRB
@RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5497*
output_shapes
: *R
then_branchCRA
?RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5496╙
>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentity>RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: o
-RaggedFromRowSplits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :К
.RaggedFromRowSplits/assert_rank_at_least/ShapeShapeRegexSplitWithOffsets:tokens:0*
T0*
_output_shapes
::э╧u
WRaggedFromRowSplits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 О
HRaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 с
&RaggedFromRowSplits/control_dependencyIdentityIRaggedFromRowSplits/RowPartitionFromRowSplits/control_dependency:output:0?^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityI^RaggedFromRowSplits/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         В
@RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :б
ARaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
=RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_1/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = ░
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:output:06^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5537*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5536Л
ZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
?RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_1/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5573*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5572╣
qRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ∙
BRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         О
RaggedFromRowSplits_1/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_1/strided_sliceStridedSlice$RaggedFromRowSplits_1/Shape:output:02RaggedFromRowSplits_1/strided_slice/stack:output:04RaggedFromRowSplits_1/strided_slice/stack_1:output:04RaggedFromRowSplits_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_1/strided_slice_1StridedSliceKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_1/strided_slice_1/stack:output:06RaggedFromRowSplits_1/strided_slice_1/stack_1:output:06RaggedFromRowSplits_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_1/assert_equal_1/EqualEqual.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_1/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_1/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_1/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_1/assert_equal_1/rangeRange9RaggedFromRowSplits_1/assert_equal_1/range/start:output:02RaggedFromRowSplits_1/assert_equal_1/Rank:output:09RaggedFromRowSplits_1/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_1/assert_equal_1/AllAll.RaggedFromRowSplits_1/assert_equal_1/Equal:z:03RaggedFromRowSplits_1/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_1/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_1/strided_slice_1:0) = а
3RaggedFromRowSplits_1/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_1/strided_slice:0) = ╖
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_1/assert_equal_1/All:output:01RaggedFromRowSplits_1/assert_equal_1/All:output:0.RaggedFromRowSplits_1/strided_slice_1:output:0,RaggedFromRowSplits_1/strided_slice:output:0i^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5610*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5609╫
@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_1/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :У
0RaggedFromRowSplits_1/assert_rank_at_least/ShapeShape%RegexSplitWithOffsets:begin_offsets:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_1/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 щ
(RaggedFromRowSplits_1/control_dependencyIdentityKRaggedFromRowSplits_1/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_1/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         В
@RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :б
ARaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/ShapeShape"RegexSplitWithOffsets:row_splits:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
=RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_sliceStridedSlice"RegexSplitWithOffsets:row_splits:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_2/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:0) = ▓
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_2/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5650*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5649Л
ZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:▓
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
?RaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2StridedSlice"RegexSplitWithOffsets:row_splits:0NRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_2/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_2/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_2/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5686*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_2_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5685╣
qRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ∙
BRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependencyIdentity"RegexSplitWithOffsets:row_splits:0[^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         М
RaggedFromRowSplits_2/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_2/strided_sliceStridedSlice$RaggedFromRowSplits_2/Shape:output:02RaggedFromRowSplits_2/strided_slice/stack:output:04RaggedFromRowSplits_2/strided_slice/stack_1:output:04RaggedFromRowSplits_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_2/strided_slice_1StridedSliceKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_2/strided_slice_1/stack:output:06RaggedFromRowSplits_2/strided_slice_1/stack_1:output:06RaggedFromRowSplits_2/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_2/assert_equal_1/EqualEqual.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_2/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_2/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_2/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_2/assert_equal_1/rangeRange9RaggedFromRowSplits_2/assert_equal_1/range/start:output:02RaggedFromRowSplits_2/assert_equal_1/Rank:output:09RaggedFromRowSplits_2/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_2/assert_equal_1/AllAll.RaggedFromRowSplits_2/assert_equal_1/Equal:z:03RaggedFromRowSplits_2/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_2/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_2/strided_slice_1:0) = а
3RaggedFromRowSplits_2/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_2/strided_slice:0) = ╖
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_2/assert_equal_1/All:output:01RaggedFromRowSplits_2/assert_equal_1/All:output:0.RaggedFromRowSplits_2/strided_slice_1:output:0,RaggedFromRowSplits_2/strided_slice:output:0i^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_false_5723*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_2_assert_equal_1_Assert_AssertGuard_true_5722╫
@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_2/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :С
0RaggedFromRowSplits_2/assert_rank_at_least/ShapeShape#RegexSplitWithOffsets:end_offsets:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_2/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 щ
(RaggedFromRowSplits_2/control_dependencyIdentityKRaggedFromRowSplits_2/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_2/assert_rank_at_least/static_checks_determined_all_ok*
T0	*(
_class
loc:@RegexSplitWithOffsets*#
_output_shapes
:         х
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsWordpieceTokenizeWithOffsetsRegexSplitWithOffsets:tokens:0iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_table*P
_output_shapes>
<:         :         :         :         *
max_bytes_per_wordd*)
output_row_partition_type
row_splits*
suffix_indicator##*
unknown_token[UNK]*
use_unknown_token(Л
QWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucketStringToHashBucketFastfWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0*#
_output_shapes
:         *
num_buckets╙
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tablefWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_valueW^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*	
Tin0*

Tout0	*#
_output_shapes
:         ▀
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2iwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_vocab_lookup_tabled^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2*
_output_shapes
: ╞
IWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/AddAddV2ZWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/hash_bucket:output:0hWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*#
_output_shapes
:         щ
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqualNotEquallWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0qwordpiecetokenizewithoffsets_wordpiecetokenizewithoffsets_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*#
_output_shapes
:         Щ
NWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2SelectV2RWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/NotEqual:z:0lWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2:values:0MWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/Add:z:0*
T0	*#
_output_shapes
:         O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы
GatherV2GatherV2kWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:output_row_lengths:0/RaggedFromRowSplits/control_dependency:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:         В
@RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/rankConst*
_output_shapes
: *
dtype0*
value	B :Р
ARaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/ShapeShapeGatherV2:output:0*
T0	*
_output_shapes
::э╧И
jRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 б
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 Н
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
=RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_sliceStridedSliceGatherV2:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_1:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskw
5RaggedFromRowSplits_3/RowPartitionFromRowSplits/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ў
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/EqualEqualFRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:0*
T0	*
_output_shapes
: Е
CRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : М
JRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :▀
DRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/rangeRangeSRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/start:output:0LRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Rank:output:0SRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range/delta:output:0*
_output_shapes
: ·
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/AllAllHRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Equal:z:0MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/range:output:0*
_output_shapes
: ╬
KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBArguments to from_row_splits do not form a valid RaggedTensor:zero╣
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:╘
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:0) = ╠
MRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*O
valueFBD B>y (RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:0) = ▓
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardIfKRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0KRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/All:output:0FRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice:output:0>RaggedFromRowSplits_3/RowPartitionFromRowSplits/Const:output:08^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *o
else_branch`R^
\RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5777*
output_shapes
: *n
then_branch_R]
[RaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5776Л
ZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/IdentityIdentityZRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: П
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskП
ERaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         С
GRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
?RaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2StridedSliceGatherV2:output:0NRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_1:output:0PRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask№
3RaggedFromRowSplits_3/RowPartitionFromRowSplits/subSubHRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_1:output:0HRaggedFromRowSplits_3/RowPartitionFromRowSplits/strided_slice_2:output:0*
T0	*#
_output_shapes
:         Л
IRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R з
_RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual	LessEqualRRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/Const:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0*
T0	*#
_output_shapes
:         е
[RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/ConstConst*
_output_shapes
:*
dtype0*
valueB: ├
YRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/AllAllcRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/LessEqual:z:0dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Const:output:0*
_output_shapes
: э
bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*[
valueRBP BJArguments to from_row_splits do not form a valid RaggedTensor:monotonic.  ╨
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= 0 did not hold element-wise:с
dRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*M
valueDBB B<x (RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:0) = ё
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardIfbRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:0bRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/All:output:07RaggedFromRowSplits_3/RowPartitionFromRowSplits/sub:z:0R^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
	*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *Ж
else_branchwRu
sRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5813*
output_shapes
: *Е
then_branchvRt
rRaggedFromRowSplits_3_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5812╣
qRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/IdentityIdentityqRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: █
BRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependencyIdentityGatherV2:output:0[^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard/Identityr^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard/Identity\^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_rank/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:         └
RaggedFromRowSplits_3/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨s
)RaggedFromRowSplits_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+RaggedFromRowSplits_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+RaggedFromRowSplits_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#RaggedFromRowSplits_3/strided_sliceStridedSlice$RaggedFromRowSplits_3/Shape:output:02RaggedFromRowSplits_3/strided_slice/stack:output:04RaggedFromRowSplits_3/strided_slice/stack_1:output:04RaggedFromRowSplits_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask~
+RaggedFromRowSplits_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-RaggedFromRowSplits_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-RaggedFromRowSplits_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
%RaggedFromRowSplits_3/strided_slice_1StridedSliceKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:04RaggedFromRowSplits_3/strided_slice_1/stack:output:06RaggedFromRowSplits_3/strided_slice_1/stack_1:output:06RaggedFromRowSplits_3/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask▓
*RaggedFromRowSplits_3/assert_equal_1/EqualEqual.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0*
T0	*
_output_shapes
: k
)RaggedFromRowSplits_3/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_3/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : r
0RaggedFromRowSplits_3/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :ў
*RaggedFromRowSplits_3/assert_equal_1/rangeRange9RaggedFromRowSplits_3/assert_equal_1/range/start:output:02RaggedFromRowSplits_3/assert_equal_1/Rank:output:09RaggedFromRowSplits_3/assert_equal_1/range/delta:output:0*
_output_shapes
: м
(RaggedFromRowSplits_3/assert_equal_1/AllAll.RaggedFromRowSplits_3/assert_equal_1/Equal:z:03RaggedFromRowSplits_3/assert_equal_1/range:output:0*
_output_shapes
: │
1RaggedFromRowSplits_3/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*R
valueIBG BAArguments to _from_row_partition do not form a valid RaggedTensorЯ
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:в
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (RaggedFromRowSplits_3/strided_slice_1:0) = а
3RaggedFromRowSplits_3/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (RaggedFromRowSplits_3/strided_slice:0) = ╖
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardIf1RaggedFromRowSplits_3/assert_equal_1/All:output:01RaggedFromRowSplits_3/assert_equal_1/All:output:0.RaggedFromRowSplits_3/strided_slice_1:output:0,RaggedFromRowSplits_3/strided_slice:output:0i^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *U
else_branchFRD
BRaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_false_5850*
output_shapes
: *T
then_branchERC
ARaggedFromRowSplits_3_assert_equal_1_Assert_AssertGuard_true_5849╫
@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityIdentity@RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: q
/RaggedFromRowSplits_3/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :┼
0RaggedFromRowSplits_3/assert_rank_at_least/ShapeShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
::э╧w
YRaggedFromRowSplits_3/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 Р
JRaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_okNoOp*&
 _has_manual_control_dependencies(*
_output_shapes
 ▄
(RaggedFromRowSplits_3/control_dependencyIdentityKRaggedFromRowSplits_3/RowPartitionFromRowSplits/control_dependency:output:0A^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard/IdentityK^RaggedFromRowSplits_3/assert_rank_at_least/static_checks_determined_all_ok*
T0	*
_class
loc:@GatherV2*#
_output_shapes
:         Ц
RaggedBoundingBox/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨╛
RaggedBoundingBox/Shape_1ShapeWWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨o
%RaggedBoundingBox/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'RaggedBoundingBox/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'RaggedBoundingBox/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
RaggedBoundingBox/strided_sliceStridedSlice RaggedBoundingBox/Shape:output:0.RaggedBoundingBox/strided_slice/stack:output:00RaggedBoundingBox/strided_slice/stack_1:output:00RaggedBoundingBox/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskY
RaggedBoundingBox/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЙ
RaggedBoundingBox/subSub(RaggedBoundingBox/strided_slice:output:0 RaggedBoundingBox/sub/y:output:0*
T0	*
_output_shapes
: q
'RaggedBoundingBox/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)RaggedBoundingBox/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)RaggedBoundingBox/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
!RaggedBoundingBox/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_1/stack:output:02RaggedBoundingBox/strided_slice_1/stack_1:output:02RaggedBoundingBox/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskq
'RaggedBoundingBox/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
)RaggedBoundingBox/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         s
)RaggedBoundingBox/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╦
!RaggedBoundingBox/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:00RaggedBoundingBox/strided_slice_2/stack:output:02RaggedBoundingBox/strided_slice_2/stack_1:output:02RaggedBoundingBox/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_maskд
RaggedBoundingBox/sub_1Sub*RaggedBoundingBox/strided_slice_1:output:0*RaggedBoundingBox/strided_slice_2:output:0*
T0	*#
_output_shapes
:         a
RaggedBoundingBox/ConstConst*
_output_shapes
:*
dtype0*
valueB: |
RaggedBoundingBox/MaxMaxRaggedBoundingBox/sub_1:z:0 RaggedBoundingBox/Const:output:0*
T0	*
_output_shapes
: ]
RaggedBoundingBox/Maximum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R Л
RaggedBoundingBox/MaximumMaximumRaggedBoundingBox/Max:output:0$RaggedBoundingBox/Maximum/y:output:0*
T0	*
_output_shapes
: q
'RaggedBoundingBox/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)RaggedBoundingBox/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)RaggedBoundingBox/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
!RaggedBoundingBox/strided_slice_3StridedSlice"RaggedBoundingBox/Shape_1:output:00RaggedBoundingBox/strided_slice_3/stack:output:02RaggedBoundingBox/strided_slice_3/stack_1:output:02RaggedBoundingBox/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskЗ
RaggedBoundingBox/stackPackRaggedBoundingBox/sub:z:0RaggedBoundingBox/Maximum:z:0*
N*
T0	*
_output_shapes
:_
RaggedBoundingBox/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╚
RaggedBoundingBox/concatConcatV2 RaggedBoundingBox/stack:output:0*RaggedBoundingBox/strided_slice_3:output:0&RaggedBoundingBox/concat/axis:output:0*
N*
T0	*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
strided_sliceStridedSlice!RaggedBoundingBox/concat:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskM
Fill/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 Rm
	Fill/dimsPackstrided_slice:output:0Fill/dims/1:output:0*
N*
T0	*
_output_shapes
:p
FillFillFill/dims:output:0
fill_value*
T0	*'
_output_shapes
:         *

index_type0	O
Fill_1/dims/1Const*
_output_shapes
: *
dtype0	*
value	B	 Rq
Fill_1/dimsPackstrided_slice:output:0Fill_1/dims/1:output:0*
N*
T0	*
_output_shapes
:v
Fill_1FillFill_1/dims:output:0fill_1_value*
T0	*'
_output_shapes
:         *

index_type0	~
#RaggedConcat/RaggedFromTensor/ShapeShapeFill:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨{
1RaggedConcat/RaggedFromTensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3RaggedConcat/RaggedFromTensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3RaggedConcat/RaggedFromTensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+RaggedConcat/RaggedFromTensor/strided_sliceStridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0:RaggedConcat/RaggedFromTensor/strided_slice/stack:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_1:output:0<RaggedConcat/RaggedFromTensor/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask}
3RaggedConcat/RaggedFromTensor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_1StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_1/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask}
3RaggedConcat/RaggedFromTensor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_2StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_2/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask╣
!RaggedConcat/RaggedFromTensor/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_1:output:06RaggedConcat/RaggedFromTensor/strided_slice_2:output:0*
T0	*
_output_shapes
: }
3RaggedConcat/RaggedFromTensor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
-RaggedConcat/RaggedFromTensor/strided_slice_3StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_3/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskК
-RaggedConcat/RaggedFromTensor/concat/values_0Pack%RaggedConcat/RaggedFromTensor/mul:z:0*
N*
T0	*
_output_shapes
:k
)RaggedConcat/RaggedFromTensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : В
$RaggedConcat/RaggedFromTensor/concatConcatV26RaggedConcat/RaggedFromTensor/concat/values_0:output:06RaggedConcat/RaggedFromTensor/strided_slice_3:output:02RaggedConcat/RaggedFromTensor/concat/axis:output:0*
N*
T0	*
_output_shapes
:к
%RaggedConcat/RaggedFromTensor/ReshapeReshapeFill:output:0-RaggedConcat/RaggedFromTensor/concat:output:0*
Tshape0	*
T0	*#
_output_shapes
:         }
3RaggedConcat/RaggedFromTensor/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:я
-RaggedConcat/RaggedFromTensor/strided_slice_4StridedSlice,RaggedConcat/RaggedFromTensor/Shape:output:0<RaggedConcat/RaggedFromTensor/strided_slice_4/stack:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_1:output:0>RaggedConcat/RaggedFromTensor/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maske
#RaggedConcat/RaggedFromTensor/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R║
>RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/ShapeShape.RaggedConcat/RaggedFromTensor/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Ц
LRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ш
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ш
NRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
FRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_sliceStridedSliceGRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/Shape:output:0URaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_1:output:0WRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskб
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rй
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV26RaggedConcat/RaggedFromTensor/strided_slice_4:output:0hRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: з
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R з
eRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 R▄
_RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangenRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0nRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         к
]RaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMulhRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0,RaggedConcat/RaggedFromTensor/Const:output:0*
T0	*#
_output_shapes
:         В
%RaggedConcat/RaggedFromTensor_1/ShapeShapeFill_1:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨}
3RaggedConcat/RaggedFromTensor_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
-RaggedConcat/RaggedFromTensor_1/strided_sliceStridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0<RaggedConcat/RaggedFromTensor_1/strided_slice/stack:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_1:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
5RaggedConcat/RaggedFromTensor_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_1StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask
5RaggedConcat/RaggedFromTensor_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_2StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask┐
#RaggedConcat/RaggedFromTensor_1/mulMul8RaggedConcat/RaggedFromTensor_1/strided_slice_1:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_2:output:0*
T0	*
_output_shapes
: 
5RaggedConcat/RaggedFromTensor_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:є
/RaggedConcat/RaggedFromTensor_1/strided_slice_3StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_3/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
end_maskО
/RaggedConcat/RaggedFromTensor_1/concat/values_0Pack'RaggedConcat/RaggedFromTensor_1/mul:z:0*
N*
T0	*
_output_shapes
:m
+RaggedConcat/RaggedFromTensor_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : К
&RaggedConcat/RaggedFromTensor_1/concatConcatV28RaggedConcat/RaggedFromTensor_1/concat/values_0:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_3:output:04RaggedConcat/RaggedFromTensor_1/concat/axis:output:0*
N*
T0	*
_output_shapes
:░
'RaggedConcat/RaggedFromTensor_1/ReshapeReshapeFill_1:output:0/RaggedConcat/RaggedFromTensor_1/concat:output:0*
Tshape0	*
T0	*#
_output_shapes
:         
5RaggedConcat/RaggedFromTensor_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
/RaggedConcat/RaggedFromTensor_1/strided_slice_4StridedSlice.RaggedConcat/RaggedFromTensor_1/Shape:output:0>RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_1:output:0@RaggedConcat/RaggedFromTensor_1/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskg
%RaggedConcat/RaggedFromTensor_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R╛
@RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/ShapeShape0RaggedConcat/RaggedFromTensor_1/Reshape:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨Ш
NRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ъ
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ъ
PRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
HRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_sliceStridedSliceIRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/Shape:output:0WRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_1:output:0YRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskг
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rп
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/addAddV28RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0jRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add/y:output:0*
T0	*
_output_shapes
: й
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/startConst*
_output_shapes
: *
dtype0	*
value	B	 R й
gRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/deltaConst*
_output_shapes
: *
dtype0	*
value	B	 Rф
aRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/rangeRangepRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/start:output:0cRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/add:z:0pRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range/delta:output:0*

Tidx0	*#
_output_shapes
:         ░
_RaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mulMuljRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/range:output:0.RaggedConcat/RaggedFromTensor_1/Const:output:0*
T0	*#
_output_shapes
:         Я
 RaggedConcat/RaggedNRows_1/ShapeShape1RaggedFromRowSplits_3/control_dependency:output:0*
T0	*
_output_shapes
:*
out_type0	:э╨x
.RaggedConcat/RaggedNRows_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0RaggedConcat/RaggedNRows_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0RaggedConcat/RaggedNRows_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(RaggedConcat/RaggedNRows_1/strided_sliceStridedSlice)RaggedConcat/RaggedNRows_1/Shape:output:07RaggedConcat/RaggedNRows_1/strided_slice/stack:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_1:output:09RaggedConcat/RaggedNRows_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskb
 RaggedConcat/RaggedNRows_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rд
RaggedConcat/RaggedNRows_1/subSub1RaggedConcat/RaggedNRows_1/strided_slice:output:0)RaggedConcat/RaggedNRows_1/sub/y:output:0*
T0	*
_output_shapes
: з
!RaggedConcat/assert_equal_1/EqualEqual6RaggedConcat/RaggedFromTensor/strided_slice_4:output:0"RaggedConcat/RaggedNRows_1/sub:z:0*
T0	*
_output_shapes
: b
 RaggedConcat/assert_equal_1/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_1/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_1/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╙
!RaggedConcat/assert_equal_1/rangeRange0RaggedConcat/assert_equal_1/range/start:output:0)RaggedConcat/assert_equal_1/Rank:output:00RaggedConcat/assert_equal_1/range/delta:output:0*
_output_shapes
: С
RaggedConcat/assert_equal_1/AllAll%RaggedConcat/assert_equal_1/Equal:z:0*RaggedConcat/assert_equal_1/range:output:0*
_output_shapes
: л
(RaggedConcat/assert_equal_1/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 1 (=y) have incompatible shapes.Ц
*RaggedConcat/assert_equal_1/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:б
*RaggedConcat/assert_equal_1/Assert/Const_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = Т
*RaggedConcat/assert_equal_1/Assert/Const_3Const*
_output_shapes
: *
dtype0*8
value/B- B'y (RaggedConcat/RaggedNRows_1/sub:0) = ╫
.RaggedConcat/assert_equal_1/Assert/AssertGuardIf(RaggedConcat/assert_equal_1/All:output:0(RaggedConcat/assert_equal_1/All:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:0"RaggedConcat/RaggedNRows_1/sub:z:08^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_1_Assert_AssertGuard_false_6006*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_1_Assert_AssertGuard_true_6005┼
7RaggedConcat/assert_equal_1/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_1/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ╜
!RaggedConcat/assert_equal_3/EqualEqual6RaggedConcat/RaggedFromTensor/strided_slice_4:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0*
T0	*
_output_shapes
: b
 RaggedConcat/assert_equal_3/RankConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_3/range/startConst*
_output_shapes
: *
dtype0*
value	B : i
'RaggedConcat/assert_equal_3/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :╙
!RaggedConcat/assert_equal_3/rangeRange0RaggedConcat/assert_equal_3/range/start:output:0)RaggedConcat/assert_equal_3/Rank:output:00RaggedConcat/assert_equal_3/range/delta:output:0*
_output_shapes
: С
RaggedConcat/assert_equal_3/AllAll%RaggedConcat/assert_equal_3/Equal:z:0*RaggedConcat/assert_equal_3/range:output:0*
_output_shapes
: л
(RaggedConcat/assert_equal_3/Assert/ConstConst*
_output_shapes
: *
dtype0*S
valueJBH BBInput tensors at index 0 (=x) and 2 (=y) have incompatible shapes.Ц
*RaggedConcat/assert_equal_3/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:б
*RaggedConcat/assert_equal_3/Assert/Const_2Const*
_output_shapes
: *
dtype0*G
value>B< B6x (RaggedConcat/RaggedFromTensor/strided_slice_4:0) = г
*RaggedConcat/assert_equal_3/Assert/Const_3Const*
_output_shapes
: *
dtype0*I
value@B> B8y (RaggedConcat/RaggedFromTensor_1/strided_slice_4:0) = ф
.RaggedConcat/assert_equal_3/Assert/AssertGuardIf(RaggedConcat/assert_equal_3/All:output:0(RaggedConcat/assert_equal_3/All:output:06RaggedConcat/RaggedFromTensor/strided_slice_4:output:08RaggedConcat/RaggedFromTensor_1/strided_slice_4:output:0/^RaggedConcat/assert_equal_1/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *L
else_branch=R;
9RaggedConcat_assert_equal_3_Assert_AssertGuard_false_6036*
output_shapes
: *K
then_branch<R:
8RaggedConcat_assert_equal_3_Assert_AssertGuard_true_6035┼
7RaggedConcat/assert_equal_3/Assert/AssertGuard/IdentityIdentity7RaggedConcat/assert_equal_3/Assert/AssertGuard:output:0*
T0
*&
 _has_manual_control_dependencies(*
_output_shapes
: ╬
RaggedConcat/concat/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ┤
RaggedConcat/concatConcatV2.RaggedConcat/RaggedFromTensor/Reshape:output:0WWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/SelectV2:output:00RaggedConcat/RaggedFromTensor_1/Reshape:output:0!RaggedConcat/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ч
 RaggedConcat/strided_slice/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         р
"RaggedConcat/strided_slice/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: р
"RaggedConcat/strided_slice/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╪
RaggedConcat/strided_sliceStridedSliceaRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0)RaggedConcat/strided_slice/stack:output:0+RaggedConcat/strided_slice/stack_1:output:0+RaggedConcat/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskр
"RaggedConcat/strided_slice_1/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
$RaggedConcat/strided_slice_1/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_1/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╡
RaggedConcat/strided_slice_1StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_1/stack:output:0-RaggedConcat/strided_slice_1/stack_1:output:0-RaggedConcat/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskУ
RaggedConcat/addAddV2%RaggedConcat/strided_slice_1:output:0#RaggedConcat/strided_slice:output:0*
T0	*#
_output_shapes
:         щ
"RaggedConcat/strided_slice_2/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         т
$RaggedConcat/strided_slice_2/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_2/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:░
RaggedConcat/strided_slice_2StridedSlice1RaggedFromRowSplits_3/control_dependency:output:0+RaggedConcat/strided_slice_2/stack:output:0-RaggedConcat/strided_slice_2/stack_1:output:0-RaggedConcat/strided_slice_2/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_maskИ
RaggedConcat/add_1AddV2#RaggedConcat/strided_slice:output:0%RaggedConcat/strided_slice_2:output:0*
T0	*
_output_shapes
: р
"RaggedConcat/strided_slice_3/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
$RaggedConcat/strided_slice_3/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_3/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:ч
RaggedConcat/strided_slice_3StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_3/stack:output:0-RaggedConcat/strided_slice_3/stack_1:output:0-RaggedConcat/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *
end_maskИ
RaggedConcat/add_2AddV2%RaggedConcat/strided_slice_3:output:0RaggedConcat/add_1:z:0*
T0	*#
_output_shapes
:         щ
"RaggedConcat/strided_slice_4/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         т
$RaggedConcat/strided_slice_4/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_4/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:т
RaggedConcat/strided_slice_4StridedSlicecRaggedConcat/RaggedFromTensor_1/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0+RaggedConcat/strided_slice_4/stack:output:0-RaggedConcat/strided_slice_4/stack_1:output:0-RaggedConcat/strided_slice_4/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask{
RaggedConcat/add_3AddV2RaggedConcat/add_1:z:0%RaggedConcat/strided_slice_4:output:0*
T0	*
_output_shapes
: ╨
RaggedConcat/concat_1/axisConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : О
RaggedConcat/concat_1ConcatV2aRaggedConcat/RaggedFromTensor/RaggedFromUniformRowLength/RowPartitionFromUniformRowLength/mul:z:0RaggedConcat/add:z:0RaggedConcat/add_2:z:0#RaggedConcat/concat_1/axis:output:0*
N*
T0	*#
_output_shapes
:         ╚
RaggedConcat/mul/yConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 RН
RaggedConcat/mulMul6RaggedConcat/RaggedFromTensor/strided_slice_4:output:0RaggedConcat/mul/y:output:0*
T0	*
_output_shapes
: ╬
RaggedConcat/range/startConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : ╬
RaggedConcat/range/deltaConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B :r
RaggedConcat/range/CastCast!RaggedConcat/range/start:output:0*

DstT0	*

SrcT0*
_output_shapes
: t
RaggedConcat/range/Cast_1Cast!RaggedConcat/range/delta:output:0*

DstT0	*

SrcT0*
_output_shapes
: Ю
RaggedConcat/rangeRangeRaggedConcat/range/Cast:y:0RaggedConcat/mul:z:0RaggedConcat/range/Cast_1:y:0*

Tidx0	*#
_output_shapes
:         ▀
RaggedConcat/Reshape/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       У
RaggedConcat/ReshapeReshapeRaggedConcat/range:output:0#RaggedConcat/Reshape/shape:output:0*
T0	*'
_output_shapes
:         р
RaggedConcat/transpose/permConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB"       Ъ
RaggedConcat/transpose	TransposeRaggedConcat/Reshape:output:0$RaggedConcat/transpose/perm:output:0*
T0	*'
_output_shapes
:         у
RaggedConcat/Reshape_1/shapeConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:
         Т
RaggedConcat/Reshape_1ReshapeRaggedConcat/transpose:y:0%RaggedConcat/Reshape_1/shape:output:0*
T0	*#
_output_shapes
:         Ь
&RaggedConcat/RaggedGather/RaggedGatherRaggedGatherRaggedConcat/concat_1:output:0RaggedConcat/concat:output:0RaggedConcat/Reshape_1:output:0*
OUTPUT_RAGGED_RANK*
PARAMS_RAGGED_RANK*
Tindices0	*
Tvalues0	*2
_output_shapes 
:         :         р
"RaggedConcat/strided_slice_5/stackConst8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_5/stack_1Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB: т
$RaggedConcat/strided_slice_5/stack_2Const8^RaggedConcat/assert_equal_1/Assert/AssertGuard/Identity8^RaggedConcat/assert_equal_3/Assert/AssertGuard/Identity*
_output_shapes
:*
dtype0*
valueB:╙
RaggedConcat/strided_slice_5StridedSlice=RaggedConcat/RaggedGather/RaggedGather:output_nested_splits:0+RaggedConcat/strided_slice_5/stack:output:0-RaggedConcat/strided_slice_5/stack_1:output:0-RaggedConcat/strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:         *

begin_mask*
end_maskЗ
IdentityIdentity<RaggedConcat/RaggedGather/RaggedGather:output_dense_values:0^NoOp*
T0	*#
_output_shapes
:         r

Identity_1Identity%RaggedConcat/strided_slice_5:output:0^NoOp*
T0	*#
_output_shapes
:         Е
NoOpNoOp/^RaggedConcat/assert_equal_1/Assert/AssertGuard/^RaggedConcat/assert_equal_3/Assert/AssertGuardP^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardg^RaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard6^RaggedFromRowSplits/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuardR^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardi^RaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard8^RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuardd^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2b^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2W^WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : : : 2`
.RaggedConcat/assert_equal_1/Assert/AssertGuard.RaggedConcat/assert_equal_1/Assert/AssertGuard2`
.RaggedConcat/assert_equal_3/Assert/AssertGuard.RaggedConcat/assert_equal_3/Assert/AssertGuard2в
ORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardORaggedFromRowSplits/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╨
fRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardfRaggedFromRowSplits/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2n
5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard5RaggedFromRowSplits/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_1/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_1/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_2/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_2/assert_equal_1/Assert/AssertGuard2ж
QRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuardQRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_equal_1/Assert/AssertGuard2╘
hRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuardhRaggedFromRowSplits_3/RowPartitionFromRowSplits/assert_non_negative/assert_less_equal/Assert/AssertGuard2r
7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard7RaggedFromRowSplits_3/assert_equal_1/Assert/AssertGuard2╩
cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV2cWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Lookup/LookupTableFindV22╞
aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV2aWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/None_Lookup/None_Size/LookupTableSizeV22░
VWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsetsVWordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets/WordpieceTokenizeWithOffsets:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :2.
,
_user_specified_namevocab_lookup_table:L H
#
_output_shapes
:         
!
_user_specified_name	strings"эL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp2*

asset_path_initializer:0vocab.vi.txt2,

asset_path_initializer_1:0vocab.ja.txt2,

asset_path_initializer_2:0vocab.vi.txt2,

asset_path_initializer_3:0vocab.ja.txt:Сa
>
ja
vi

signatures"
_generic_user_object
╩
	tokenizer
_reserved_tokens
_vocab_path
	vocab

detokenize
	get_reserved_tokens

get_vocab_path
get_vocab_size

lookup
tokenize"
_generic_user_object
╩
	tokenizer
_reserved_tokens
_vocab_path
	vocab

detokenize
get_reserved_tokens
get_vocab_path
get_vocab_size

lookup
tokenize"
_generic_user_object
"
signature_map
N
_basic_tokenizer
_wordpiece_tokenizer"
_generic_user_object
 "
trackable_list_wrapper
*
:░ъ2Variable
П
trace_0
trace_12╪
__inference_detokenize_3471
__inference_detokenize_3838Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
╒
trace_02╕
$__inference_get_reserved_tokens_3842П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ztrace_0
╨
trace_02│
__inference_get_vocab_path_3847П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ztrace_0
╨
trace_02│
__inference_get_vocab_size_3857П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ztrace_0
З
trace_0
 trace_12╨
__inference_lookup_3863
__inference_lookup_3873Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0z trace_1
╘
!trace_02╖
__inference_tokenize_4584Щ
Т▓О
FullArgSpec
argsЪ
	jstrings
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z!trace_0
N
"_basic_tokenizer
#_wordpiece_tokenizer"
_generic_user_object
 "
trackable_list_wrapper
*
:░ъ2Variable
П
$trace_0
%trace_12╪
__inference_detokenize_4991
__inference_detokenize_5358Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z$trace_0z%trace_1
╒
&trace_02╕
$__inference_get_reserved_tokens_5362П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z&trace_0
╨
'trace_02│
__inference_get_vocab_path_5367П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z'trace_0
╨
(trace_02│
__inference_get_vocab_size_5377П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z(trace_0
З
)trace_0
*trace_12╨
__inference_lookup_5383
__inference_lookup_5393Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z)trace_0z*trace_1
╘
+trace_02╖
__inference_tokenize_6104Щ
Т▓О
FullArgSpec
argsЪ
	jstrings
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z+trace_0
"
_generic_user_object
7
,_vocab_lookup_table"
_generic_user_object
╦B╚
__inference_detokenize_3471	tokenized"Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪B╒
__inference_detokenize_3838	tokenizedtokenized_1"Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╗B╕
$__inference_get_reserved_tokens_3842"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╘
	capture_0B│
__inference_get_vocab_path_3847"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z	capture_0
╢B│
__inference_get_vocab_size_3857"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╟B─
__inference_lookup_3863	token_ids"Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘B╤
__inference_lookup_3873	token_idstoken_ids_1"Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Я
-	capture_1
.	capture_2
/	capture_3B┬
__inference_tokenize_4584strings"Щ
Т▓О
FullArgSpec
argsЪ
	jstrings
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z-	capture_1z.	capture_2z/	capture_3
"
_generic_user_object
7
0_vocab_lookup_table"
_generic_user_object
╦B╚
__inference_detokenize_4991	tokenized"Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪B╒
__inference_detokenize_5358	tokenizedtokenized_1"Ы
Ф▓Р
FullArgSpec
argsЪ
j	tokenized
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╗B╕
$__inference_get_reserved_tokens_5362"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╘
	capture_0B│
__inference_get_vocab_path_5367"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z	capture_0
╢B│
__inference_get_vocab_size_5377"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╟B─
__inference_lookup_5383	token_ids"Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘B╤
__inference_lookup_5393	token_idstoken_ids_1"Ы
Ф▓Р
FullArgSpec
argsЪ
j	token_ids
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Я
1	capture_1
.	capture_2
/	capture_3B┬
__inference_tokenize_6104strings"Щ
Т▓О
FullArgSpec
argsЪ
	jstrings
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z1	capture_1z.	capture_2z/	capture_3
R
2_initializer
3_create_resource
4_initialize
5_destroy_resourceR 
!J	
Const_3jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
R
6_initializer
7_create_resource
8_initialize
9_destroy_resourceR 
!J	
Const_2jtf.TrackableConstant
-
:	_filename"
_generic_user_object
╩
;trace_02н
__inference__creator_6108П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z;trace_0
╬
<trace_02▒
__inference__initializer_6114П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z<trace_0
╠
=trace_02п
__inference__destroyer_6118П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z=trace_0
-
>	_filename"
_generic_user_object
╩
?trace_02н
__inference__creator_6122П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z?trace_0
╬
@trace_02▒
__inference__initializer_6128П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z@trace_0
╠
Atrace_02п
__inference__destroyer_6132П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в zAtrace_0
*
░Bн
__inference__creator_6108"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╥
:	capture_0B▒
__inference__initializer_6114"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z:	capture_0
▓Bп
__inference__destroyer_6118"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
* 
░Bн
__inference__creator_6122"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╥
>	capture_0B▒
__inference__initializer_6128"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в z>	capture_0
▓Bп
__inference__destroyer_6132"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в >
__inference__creator_6108!в

в 
к "К
unknown >
__inference__creator_6122!в

в 
к "К
unknown @
__inference__destroyer_6118!в

в 
к "К
unknown @
__inference__destroyer_6132!в

в 
к "К
unknown F
__inference__initializer_6114%:,в

в 
к "К
unknown F
__inference__initializer_6128%>0в

в 
к "К
unknown ~
__inference_detokenize_3471_,;в8
1в.
,К)
	tokenized                  	
к "К
unknown         Э
__inference_detokenize_3838~,ZвW
PвM
KТH0в-
·                  
А	
`
А	RaggedTensorSpec 
к "К
unknown         ~
__inference_detokenize_4991_0;в8
1в.
,К)
	tokenized                  	
к "К
unknown         Э
__inference_detokenize_5358~0ZвW
PвM
KТH0в-
·                  
А	
`
А	RaggedTensorSpec 
к "К
unknown         M
$__inference_get_reserved_tokens_3842%в

в 
к "К
unknownM
$__inference_get_reserved_tokens_5362%в

в 
к "К
unknownG
__inference_get_vocab_path_3847$в

в 
к "К
unknown G
__inference_get_vocab_path_5367$в

в 
к "К
unknown G
__inference_get_vocab_size_3857$в

в 
к "К
unknown G
__inference_get_vocab_size_5377$в

в 
к "К
unknown З
__inference_lookup_3863l;в8
1в.
,К)
	token_ids                  	
к "*К'
unknown                  ╚
__inference_lookup_3873мZвW
PвM
KТH0в-
·                  
А	
`
А	RaggedTensorSpec 
к "KТH0в-
·                  
А
`
А	RaggedTensorSpec З
__inference_lookup_5383l;в8
1в.
,К)
	token_ids                  	
к "*К'
unknown                  ╚
__inference_lookup_5393мZвW
PвM
KТH0в-
·                  
А	
`
А	RaggedTensorSpec 
к "KТH0в-
·                  
А
`
А	RaggedTensorSpec Я
__inference_tokenize_4584Б,-./,в)
"в
К
strings         
к "KТH0в-
·                  
А	
`
А	RaggedTensorSpec Я
__inference_tokenize_6104Б01./,в)
"в
К
strings         
к "KТH0в-
·                  
А	
`
А	RaggedTensorSpec 