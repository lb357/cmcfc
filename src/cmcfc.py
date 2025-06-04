from pycparser import c_parser
from pycparser.c_ast import *
import re
from logger import *


DeclarationType = {
	"root": 0,
	"void": 1,
	"ptr": 2,
	"int": 3, "int32_t": 3,
	"char": 4, "uint8_t": 4,
	"float": 5,
	"double": 6
}
TypeSize = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
FracCoef = {DeclarationType["float"]: 2, DeclarationType["double"]: 4}


class CMCFCCompilerError(Exception):
	def __init__(self, *args):
		super().__init__(*args)


class CMCFCAssemblerError(Exception):
	def __init__(self, *args):
		super().__init__(*args)


class CMCFCPreprocessor(object):
	def __init__(self):
		self.std_constants = {}

	def preprocess(self, text: str, includes: dict, debug: bool = False):
		definitions = {}
		while True:
			match = re.search(r"(#include [\"\'<].*[\"\'>])", text)
			if match is None:
				break
			include = match.group(0).split(" ")
			for i in range(include.count("")):
				include.remove("")
			text = text[:match.start()] + includes[include[-1]] + text[match.end():]
		while True:
			match = re.search(r"(#define .* .*)", text)
			if match is None:
				break
			definition = match.group(0).split(" ")
			for i in range(definition.count("")):
				definition.remove("")
			dfrom, dto = definition[-2:]
			definitions[dfrom] = dto
			text = text[:match.start()] + text[match.end():]

		for definition in definitions:
			text = text.replace(definition, definitions[definition])

		while True:
			match = re.search(r"//.*\n", text)
			if match is None:
				break
			text = text[:match.start()] + text[match.end():]


		if debug:
			logger.debug(f"Preprocessed C Code:\n{text}")
		return text


class CMCFCCompiler(object):
	def __init__(self, mc_namespace: str = "cmcfc_default"):
		self.sizes = {}
		self.array_sizes = {}
		self.declarations = {}
		self.ptr_types = {}
		self.function_args = {}
		self.hierarchy = {}
		self.ptr_links = {}
		self.dependencies = {}
		self.declaration_type = DeclarationType.copy()
		self.type_size = TypeSize.copy()
		self.out = {}
		self.gen_index = 0
		self.root = -1
		self.setup = -1
		self.loop = -1

		self.mc_namespace = mc_namespace
		self.reset_generator()

	def reset_generator(self):
		self.sizes = {}
		self.array_sizes = {}
		self.declarations = {}
		self.ptr_types = {}
		self.function_args = {}
		self.hierarchy = {}
		self.dependencies = {}
		self.declaration_type = DeclarationType.copy()
		self.type_size = TypeSize.copy()
		self.out = {}
		self.ptr_links = {}
		self.gen_index = 0
		self.root = self.declare(self.declaration_type["root"])

	def declare(self, declaration_type):
		if len(self.declarations) == 0:
			declaration_code = 0
		else:
			declaration_code = max(self.declarations.keys()) + 1
		self.declarations[declaration_code] = declaration_type
		self.sizes[declaration_code] = self.type_size[declaration_type]
		return declaration_code

	def add_function_arg(self, decl_code, arg_code):
		self.function_args[decl_code].append(arg_code)

	def get_gen_name(self):
		gen_index = self.gen_index
		self.gen_index += 1
		return f"gen{gen_index}"

	def gen_declcode(self, decl_type, parent_declcode, decl_like=None):
		declname = self.get_gen_name()
		declcode = self.declare(decl_type)
		self.add_hierarchy_dependency(parent_declcode, declcode, declname)
		if decl_like is not None:
			# self.sizes[declcode] = decllike
			if decl_type == self.declaration_type["ptr"]:
				self.ptr_types[declcode] = self.ptr_types[decl_like]
		return declcode

	def add_hierarchy_dependency(self, parent_declcode, child_declcode, child_name):
		if parent_declcode not in self.hierarchy:
			self.hierarchy[parent_declcode] = {}
			self.out[parent_declcode] = []
		self.hierarchy[parent_declcode][child_name] = child_declcode

	def add_space_dependency(self, parent_declcode, child_declcode):
		self.function_args[child_declcode] = []
		self.dependencies[child_declcode] = parent_declcode
		self.out[child_declcode] = []

	def get_hierarchy_space_declarations(self, declcode, hierarchy_declarations=None):
		if hierarchy_declarations is None:
			hierarchy_declarations = {}
		if declcode not in self.hierarchy:
			self.hierarchy[declcode] = {}
		hierarchy_declarations = (self.hierarchy[declcode]) | hierarchy_declarations
		if declcode in self.dependencies:
			parent_declcode = self.dependencies[declcode]
			return self.get_hierarchy_space_declarations(parent_declcode, hierarchy_declarations)
		else:
			return hierarchy_declarations

	def out_operation(self, space: int, declcode: int, operand: str, value_declcode: int, targetcode: int):
		if self.declarations[declcode] != self.declarations[value_declcode]:
			value_declcode = self.implicit_cast(value_declcode, declcode, parent_declcode=space)
		self.out[space].append(("OPERATION", operand, declcode, value_declcode, targetcode))

	def out_equating(self, space: int, declcode: int, value_declcode: int):
		if self.declarations[declcode] != self.declarations[value_declcode]:
			value_declcode = self.implicit_cast(value_declcode, declcode, parent_declcode=space)
		self.out[space].append(("EQUATING", declcode, value_declcode))

	def out_set(self, space: int, declcode: int, value: str):
		self.out[space].append(("SET", declcode, value))

	def out_condition(self, space: int, condition_declcode: int,
					  true_function_declcode: [int, None],
					  false_function_declcode: [int, None]):
		self.out[space].append(("CONDITION", condition_declcode, true_function_declcode, false_function_declcode))

	def out_call(self, space: int, function_declcode: int):
		self.out[space].append(("CALL", function_declcode))

	def out_return(self, space: int, declcode: int):
		self.out[space].append(("RETURN", declcode))

	def out_address(self, space: int, ptr_declcode: int, declcode: int):
		self.out[space].append(("ADDRESS", ptr_declcode, declcode))

	def out_dereference(self, space: int, declcode: int, ptr_declcode: int):
		self.out[space].append(("DEREFERENCE", declcode, ptr_declcode))

	def out_asm(self, space: int, asm_text: str, asm_out: [int, None], args: tuple, format: bool = True):
		text = asm_text.format(*args) if format else asm_text
		self.out[space].append(("ASM", text, asm_out))

	def out_cast(self, space: int, to_declcode, from_declcode):
		self.out[space].append(("CAST", to_declcode, from_declcode))

	def out_ptreq(self, space: int, ptr_declcode: int, value_declcode: int):
		self.out[space].append(("PTREQ", ptr_declcode, value_declcode))

	def compile(self, c_code: str, debug: bool = False) -> [dict, dict, list]:
		parser = c_parser.CParser()
		ast = parser.parse(c_code)
		if debug:
			logger.debug(f"AST:\n{ast}")
		self.reset_generator()
		self.visit(ast, self.root)
		if debug:
			logger.debug(f"DECLARATION TYPE: {self.declaration_type}")
			logger.debug(f"TYPE SIZE: {self.type_size}")
			logger.debug(f"Hierarchy: {self.hierarchy}")
			logger.debug(f"Declarations: {self.declarations}")
			logger.debug(f"Function args: {self.function_args}")
			logger.debug(f"Pointer types: {self.ptr_types}")
			logger.debug(f"Sizes: {self.sizes}")
			logger.debug(f"Ptr links: {self.ptr_links}")
			logger.debug(f"Array sizes: {self.array_sizes}")
			asm_debug_out = ""
			for func in self.out:
				if func == self.root:
					asm_debug_out += f"{func} (root):\n"
				elif func == self.setup:
					asm_debug_out += f"{func} (setup):\n"
				elif func == self.loop:
					asm_debug_out += f"{func} (loop):\n"
				else:
					asm_debug_out+=f"{func}:\n"

				for code in self.out[func]:
					asm_debug_out+=f"    {code}\n"
			logger.debug(f"ASM:\n{asm_debug_out}")
		return self.out, self.declarations, [self.root, self.setup, self.loop]

	def get_builtin_function_static_args(self, node: FuncCall, parent_declcode: int) -> tuple[tuple]:
		args: list[tuple] = []
		for arg in node.args.exprs:
			if isinstance(arg, Constant):
				if arg.type == "string":
					args.append((arg.value[1:-1], False))
				else:
					args.append((arg.value, False))
			elif isinstance(arg, ID):
				args.append((self.visit(arg, parent_declcode=parent_declcode), True))
			else:
				raise CMCFCCompilerError(f"{node.coord} {type(arg)} type not supported (see mcfunction syntax)")
		return tuple(args)

	def implicit_cast(self, value_declcode: int, target_type_declcode: int, parent_declcode: int):
		to_type = self.declarations[target_type_declcode]
		declcode = self.gen_declcode(to_type, parent_declcode=parent_declcode)
		self.out_cast(parent_declcode, declcode, value_declcode)
		return declcode


	def visit(self, node: Node, parent_declcode: int):
		# REFACTOR: IF ISINSTANCE -> FUNC CALL BY NODE NAME!!!   YandereDev style))
		if isinstance(node, FileAST):
			for sub_node in node.children():
				self.visit(sub_node[1], parent_declcode=parent_declcode)
		elif isinstance(node, Decl):
			declcode = self.visit(node.type, parent_declcode=parent_declcode)
			if node.name == "setup" and isinstance(node.type, FuncDecl):
				self.setup = declcode
			if node.name == "loop" and isinstance(node.type, FuncDecl):
				self.loop = declcode
			if node.init is not None:
				init = self.visit(node.init, parent_declcode=parent_declcode)
				if isinstance(init, list) or isinstance(init, tuple):
					vsize = self.type_size[self.declaration_type["ptr"]]
					for idx in init:
						if idx in self.sizes:
							vsize += self.sizes[idx]
						else:
							vsize += self.type_size[self.declarations[idx]]

					if vsize > self.sizes[declcode]:
						raise CMCFCCompilerError(f"{node.coord} Too many values")
					for idx in range(len(init)):
						self.out_equating(parent_declcode, declcode + self.type_size[self.declaration_type["ptr"]] + idx, init[idx])
				else:
					self.out_equating(parent_declcode, declcode, init)
			return declcode
		elif isinstance(node, PtrDecl):
			declcode = self.visit(node.type, parent_declcode=parent_declcode)
			self.ptr_types[declcode] = self.declarations[declcode]
			self.declarations[declcode] = self.declaration_type["ptr"]
			self.sizes[declcode] = self.type_size[self.declaration_type["ptr"]]
			return declcode
		elif isinstance(node, DeclList):
			decl_data = []
			if node.decls is not None:
				if len(node.decls) > 0:
					for block_item in node.decls:
						decl_data.append(self.visit(block_item, parent_declcode=parent_declcode))
			return decl_data
		elif isinstance(node, TypeDecl):
			decltype = self.visit(node.type, parent_declcode=parent_declcode)
			declcode = self.declare(decltype)
			self.add_hierarchy_dependency(
				parent_declcode=parent_declcode,
				child_declcode=declcode,
				child_name=node.declname
			)
			return declcode
		elif isinstance(node, ArrayDecl):
			declcode = self.visit(node.type, parent_declcode=parent_declcode)
			self.ptr_types[declcode] = self.declarations[declcode]
			self.declarations[declcode] = self.declaration_type["ptr"]

			if isinstance(node.dim, Constant):
				if node.dim.type == "int":
					array_size = int(node.dim.value)
					if declcode not in self.array_sizes:
						self.array_sizes[declcode] = []
					self.array_sizes[declcode].append(array_size)
				else:
					raise CMCFCCompilerError(f"{node.dim.coord} Expected int dim")
			else:
				raise CMCFCCompilerError(
					f"{node.dim.coord} C89 (possible future update) standard does not support non-constants arrays"
				)

			# Нужно будет проверить работоспособность при создании массивов структур
			self.sizes[declcode] = self.sizes[declcode] * array_size + self.type_size[self.declaration_type["ptr"]]
			array_elements = []
			for array_i in range(array_size):
				array_elements.append(self.gen_declcode(self.ptr_types[declcode], parent_declcode))
			self.out_set(parent_declcode, declcode, f"{declcode + 1}")  # f"{array_elements[0]}")
			return declcode

		elif isinstance(node, IdentifierType):
			return self.declaration_type[" ".join(node.names)]
		elif isinstance(node, Constant):
			if node.type == "string":
				declcode = self.gen_declcode(self.declaration_type["char"], parent_declcode)
				self.ptr_types[declcode] = self.declarations[declcode]
				self.declarations[declcode] = self.declaration_type["ptr"]
				array_size = len(node.value) - 2
				self.sizes[declcode] = self.sizes[declcode] * array_size + self.type_size[self.declaration_type["ptr"]]
				array_elements = []
				for array_i in range(0, array_size):
					array_elements.append(self.gen_declcode(self.declaration_type["char"], parent_declcode))
					self.out_set(parent_declcode, array_elements[array_i], node.value[array_i + 1])
				self.out_set(parent_declcode, declcode, f"{declcode + 1}")  # f"{array_elements[0]}")
			else:
				declcode = self.gen_declcode(self.declaration_type[node.type], parent_declcode=parent_declcode)
				self.out_set(parent_declcode, declcode, node.value)
			return declcode
		elif isinstance(node, FuncDef):
			declcode = self.visit(node.decl, parent_declcode=parent_declcode)
			if node.body is not None:
				self.visit(node.body, parent_declcode=declcode)
			self.out_return(declcode, declcode)
			return declcode

		elif isinstance(node, FuncDecl):
			declcode = self.visit(node.type, parent_declcode=parent_declcode)
			self.add_space_dependency(parent_declcode, declcode)

			if node.args is not None:
				for param in self.visit(node.args, parent_declcode=declcode):
					self.add_function_arg(declcode, param)
			return declcode

		elif isinstance(node, ParamList):
			return [self.visit(param, parent_declcode=parent_declcode) for param in node.params]
		elif isinstance(node, Compound):
			compound_data = []
			if node.block_items is not None:
				if len(node.block_items) > 0:
					for block_item in node.block_items:
						compound_data.append(self.visit(block_item, parent_declcode=parent_declcode))
			return compound_data
		elif isinstance(node, ExprList):
			expr_data = []
			if node.exprs is not None:
				if len(node.exprs) > 0:
					for expr in node.exprs:
						expr_data.append(self.visit(expr, parent_declcode=parent_declcode))
			return expr_data
		elif isinstance(node, InitList):
			init_data = []
			if node.exprs is not None:
				if len(node.exprs) > 0:
					for expr in node.exprs:
						data = self.visit(expr, parent_declcode=parent_declcode)
						if isinstance(data, list):
							init_data += data
						else:
							init_data.append(data)
			return init_data
		elif isinstance(node, If):
			cond_declcode = self.visit(node.cond, parent_declcode=parent_declcode)
			if node.iftrue is not None:
				iftrue_declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
				self.add_space_dependency(parent_declcode, iftrue_declcode)
				self.visit(node.iftrue, parent_declcode=iftrue_declcode)
			else:
				iftrue_declcode = None
			if node.iffalse is not None:
				iffalse_declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
				self.add_space_dependency(parent_declcode, iffalse_declcode)
				self.visit(node.iffalse, parent_declcode=iffalse_declcode)
			else:
				iffalse_declcode = None
			self.out_condition(parent_declcode, cond_declcode, iftrue_declcode, iffalse_declcode)
			self.out_return(iftrue_declcode, iftrue_declcode)
			if iffalse_declcode is not None:
				self.out_return(iffalse_declcode, iffalse_declcode)

		elif isinstance(node, BinaryOp):
			left_declcode = self.visit(node.left, parent_declcode=parent_declcode)
			if node.op in ["<", ">", "<=", ">=", "==", "!="]:
				target_declcode = self.gen_declcode(self.declaration_type["int"], parent_declcode=parent_declcode,
													decl_like=left_declcode)
			else:
				target_declcode = self.gen_declcode(self.declarations[left_declcode], parent_declcode=parent_declcode,
													decl_like=left_declcode)
			right_declcode = self.visit(node.right, parent_declcode=parent_declcode)
			#right_declcode = self.implicit_cast(right_declcode, left_declcode, parent_declcode=parent_declcode)
			self.out_operation(parent_declcode, left_declcode, node.op, right_declcode, target_declcode)
			if isinstance(node.left, ArrayRef):
				self.out_ptreq(parent_declcode, self.ptr_links[left_declcode], left_declcode)
			return target_declcode

		elif isinstance(node, UnaryOp):
			if node.op in ["p++", "p--"]:
				target_declcode = self.visit(node.expr, parent_declcode=parent_declcode)
				#value_declcode = self.gen_declcode(self.declaration_type["int"], parent_declcode=parent_declcode)
				value_declcode = self.gen_declcode(self.declarations[target_declcode], parent_declcode=parent_declcode)
				self.out_set(parent_declcode, value_declcode, "1")
				self.out_operation(parent_declcode, target_declcode, node.op[2:], value_declcode, target_declcode)
			elif node.op == "&":
				# target_declcode = self.gen_declcode(self.declaration_type["ptr"], parent_declcode=parent_declcode)
				# value_declcode = self.visit(node.expr, parent_declcode=parent_declcode)
				# self.ptr_types[target_declcode] = self.declarations[value_declcode]
				##self.sizes[target_declcode] = self.type_size[self.declaration_type["ptr"]]
				# self.out_set(parent_declcode, target_declcode, f"{value_declcode}")
				target_declcode = self.gen_declcode(self.declaration_type["ptr"], parent_declcode=parent_declcode)
				value_declcode = self.visit(node.expr, parent_declcode=parent_declcode)
				self.ptr_types[target_declcode] = self.declarations[value_declcode]
				self.out_address(parent_declcode, target_declcode, value_declcode)
			elif node.op == "*":
				value_declcode = self.visit(node.expr, parent_declcode=parent_declcode)
				target_declcode = self.gen_declcode(self.ptr_types[value_declcode], parent_declcode=parent_declcode)
				self.out_dereference(parent_declcode, target_declcode, value_declcode)
				pass
			else:
				raise CMCFCCompilerError(f"{node.coord} Unknown UnaryOp {node.op}")
			return target_declcode
		elif isinstance(node, ArrayRef):
			# It is necessary to refactor (see self.array_sizes) in order to process multidimensional arrays and struct arrays
			if isinstance(node.name, ArrayRef):
				raise CMCFCCompilerError(f"{node.coord} Multidimensional arrays cannot be processed (expected in future)")
			else:
				array_declcode = self.visit(node.name, parent_declcode=parent_declcode)
				idx_declcode = self.visit(node.subscript, parent_declcode=parent_declcode)
				sarray_declcode = self.gen_declcode(self.declarations[array_declcode], parent_declcode=parent_declcode,
													decl_like=array_declcode)
				declcode = self.gen_declcode(self.ptr_types[sarray_declcode], parent_declcode=parent_declcode)
				self.out_equating(parent_declcode, sarray_declcode, array_declcode)
				self.out_operation(parent_declcode, sarray_declcode, "+", idx_declcode, sarray_declcode)
				self.out_dereference(parent_declcode, declcode, sarray_declcode)
				self.ptr_links[declcode] = sarray_declcode
				return declcode

		elif isinstance(node, Assignment):
			left_declcode = self.visit(node.lvalue, parent_declcode=parent_declcode)
			right_declcode = self.visit(node.rvalue, parent_declcode=parent_declcode)
			target_declcode = -1

			#right_declcode = self.implicit_cast(right_declcode, left_declcode, parent_declcode=parent_declcode)

			if node.op == "=":
				target_declcode = left_declcode
				self.out_equating(parent_declcode, left_declcode, right_declcode)
			elif node.op[-1] == "=":
				target_declcode = left_declcode
				self.out_operation(parent_declcode, left_declcode, node.op[:-1], right_declcode, left_declcode)
			else:
				target_declcode = self.gen_declcode(self.declarations[left_declcode], parent_declcode=parent_declcode)
				self.out_operation(parent_declcode, left_declcode, node.op, right_declcode, target_declcode)
			if isinstance(node.lvalue, ArrayRef):
				self.out_ptreq(parent_declcode, self.ptr_links[left_declcode], left_declcode)
			return target_declcode

		elif isinstance(node, ID):
			return self.get_hierarchy_space_declarations(parent_declcode)[node.name]
		elif isinstance(node, FuncCall):
			if isinstance(node.name, ID):
				if node.name.name == "asm":
					declcode = self.gen_declcode(self.declaration_type["int"], parent_declcode=parent_declcode)
					args = self.get_builtin_function_static_args(node, parent_declcode)
					self.out_asm(parent_declcode, args[0][0], declcode, tuple([arg[0] for arg in args[1:]]), True)
					return declcode
				elif node.name.name == "printf":
					declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
					args = self.get_builtin_function_static_args(node, parent_declcode)
					raw = args[0][0].split("%s")
					color = '"gray"'
					text = "["
					for text_segment in range(len(raw)-1):
						text += '{"text":"'+str(raw[text_segment])+'","color":'+color+'},'
						if args[1+text_segment][1]:
							text += '{"storage":"'+f"{self.mc_namespace}"+':cmcfc","nbt":"'+str(args[1+text_segment][0])+'","color":'+color+'},'
						else:
							text += '{"text":"'+str(args[1+text_segment][0])+'","color":'+color+'},'
					text += '{"text":"' + str(raw[-1]) + '","color":'+color+'}'
					text += "]"

					#text = str(args[0]) % (args[1:])
					self.out_asm(parent_declcode, f'tellraw @a {text}', None, (), False)
					return declcode
				elif node.name.name in ["rand"]:
					declcode = self.gen_declcode(self.declaration_type["int"], parent_declcode=parent_declcode)
					self.out_asm(parent_declcode, f"function {self.mc_namespace}:cmcfc/{node.name.name}", declcode, (), False)
					return declcode
			declcode = self.visit(node.name, parent_declcode=parent_declcode)
			arg_values = self.visit(node.args, parent_declcode=parent_declcode)
			for arg in range(len(self.function_args[declcode])):
				self.out_equating(parent_declcode, self.function_args[declcode][arg], arg_values[arg])
			self.out_call(parent_declcode, declcode)
			return declcode
		elif isinstance(node, Return):
			declcode = self.visit(node.expr, parent_declcode=parent_declcode)
			self.out_equating(parent_declcode, parent_declcode, declcode)
			self.out_return(declcode, declcode)
			return parent_declcode
		elif isinstance(node, While):
			declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
			self.add_space_dependency(parent_declcode, declcode)
			cond_declcode = self.visit(node.cond, parent_declcode=parent_declcode)
			self.visit(node.stmt, parent_declcode=declcode)
			self.out_condition(parent_declcode, cond_declcode, declcode, None)
			recursion_cond_declcode = self.visit(node.cond, parent_declcode=declcode)
			self.out_condition(declcode, recursion_cond_declcode, declcode, None)
			self.out_return(declcode, declcode)
			return declcode
		elif isinstance(node, DoWhile):
			declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
			self.add_space_dependency(parent_declcode, declcode)
			self.visit(node.stmt, parent_declcode=parent_declcode)
			cond_declcode = self.visit(node.cond, parent_declcode=parent_declcode)
			self.visit(node.stmt, parent_declcode=declcode)
			self.out_condition(parent_declcode, cond_declcode, declcode, None)
			recursion_cond_declcode = self.visit(node.cond, parent_declcode=declcode)
			self.out_condition(declcode, recursion_cond_declcode, declcode, None)
			self.out_return(declcode, declcode)
			return declcode
		elif isinstance(node, For):
			declcode = self.gen_declcode(self.declaration_type["void"], parent_declcode=parent_declcode)
			self.add_space_dependency(parent_declcode, declcode)
			declinit_data = self.visit(node.init, parent_declcode=parent_declcode)

			cond_declcode = self.visit(node.cond, parent_declcode=parent_declcode)
			self.visit(node.stmt, parent_declcode=declcode)
			self.out_condition(parent_declcode, cond_declcode, declcode, None)
			self.visit(node.next, parent_declcode=declcode)
			recursion_cond_declcode = self.visit(node.cond, parent_declcode=declcode)
			self.out_condition(declcode, recursion_cond_declcode, declcode, None)
			self.out_return(declcode, declcode)
			return declcode
		elif isinstance(node, Typename):
			if node.name is None and isinstance(node.type, TypeDecl):
				return self.visit(node.type.type, parent_declcode=parent_declcode)
			else:
				raise CMCFCCompilerError(f"{node.coord} Unknown Typename")
		elif isinstance(node, Cast):
			to_type = self.visit(node.to_type, parent_declcode=parent_declcode)
			value_declcode = self.visit(node.expr, parent_declcode=parent_declcode)
			declcode = self.gen_declcode(to_type, parent_declcode=parent_declcode)
			self.out_cast(parent_declcode, declcode, value_declcode)
			return declcode


class CMCFCAssembler(object):
	def __init__(self, mc_namespace: str = "cmcfc_default"):
		self.mc_namespace = mc_namespace

	def assemble(self, asm_code: dict, declarations: dict, process_data: dict, debug: bool = False):
		out = {}
		root, setup, loop = process_data
		out["setup"] = f"function {self.mc_namespace}:{setup}"
		out["loop"] = f"function {self.mc_namespace}:{loop}"

		for func in asm_code:
			out[func] = ""
			if func == root:
				out[func] += f'scoreboard objectives add {self.mc_namespace}.cmcfc dummy ' + '{"text":"cmcfc",color:"gold"}\n'
				out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_ptr set value 0\n"
				out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_val set value 0\n"
				out[func] += f"data modify storage {self.mc_namespace}:cmcfc deref set value 0\n"
				out[func] += f'data modify storage {self.mc_namespace}:cmcfc namespace set value "{self.mc_namespace}:cmcfc"\n'
				for decl in declarations:
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc {decl} set value 0\n"
			elif func == setup:
				out[func] += f"function {self.mc_namespace}:{root}\n"

			for data in asm_code[func]:
				operation = data[0]
				if operation == "SET":
					declcode = data[1]
					value = data[2]
					if declarations[declcode] == DeclarationType["int"] or declarations[declcode] == DeclarationType["ptr"]:
						out[func] += f"data modify storage {self.mc_namespace}:cmcfc {declcode} set value {int(value)}\n"
					elif declarations[declcode] == DeclarationType["float"]:
						out[func] += f"data modify storage {self.mc_namespace}:cmcfc {declcode} set value {int(float(value)*100)}\n"
					elif declarations[declcode] == DeclarationType["double"]:
						out[func] += f"data modify storage {self.mc_namespace}:cmcfc {declcode} set value {int(float(value)*10000)}\n"
					else:
						raise CMCFCAssemblerError(f"Value {value} cannot be set for {declcode}")
				elif operation == "CAST":
					to_declcode = data[1]
					from_declcode = data[2]
					if declarations[to_declcode] == declarations[from_declcode] or (declarations[from_declcode] == DeclarationType["int"] and declarations[to_declcode] == DeclarationType["ptr"]):
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {to_declcode} int 1 run data get storage {self.mc_namespace}:cmcfc {from_declcode} 1\n"
					elif declarations[to_declcode] == DeclarationType["int"] and declarations[from_declcode] == DeclarationType["float"]:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {to_declcode} int {1/(10**FracCoef[DeclarationType['float']])} run data get storage {self.mc_namespace}:cmcfc {from_declcode} 1\n"
					elif declarations[to_declcode] == DeclarationType["int"] and declarations[from_declcode] == DeclarationType["double"]:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {to_declcode} int {1/(10**FracCoef[DeclarationType['double']])} run data get storage {self.mc_namespace}:cmcfc {from_declcode} 1\n"
					elif declarations[to_declcode] == DeclarationType["float"] and declarations[from_declcode] == DeclarationType["int"]:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {to_declcode} int {10**FracCoef[DeclarationType['float']]} run data get storage {self.mc_namespace}:cmcfc {from_declcode} 1\n"
					elif declarations[to_declcode] == DeclarationType["double"] and declarations[from_declcode] == DeclarationType["int"]:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {to_declcode} int {10**FracCoef[DeclarationType['double']]} run data get storage {self.mc_namespace}:cmcfc {from_declcode} 1\n"
					else:
						raise CMCFCAssemblerError(f"{from_declcode} (type {declarations[from_declcode]}) cannot be casted to {to_declcode} (type {declarations[to_declcode]})")
				elif operation == "EQUATING":
					declcode = data[1]
					value_declcode = data[2]
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc {declcode} set from storage {self.mc_namespace}:cmcfc {value_declcode}\n"
				elif operation == "OPERATION":
					operand = data[1]
					declcode = data[2]
					value_declcode = data[3]
					targetcode = data[4]

					out[func] += f"scoreboard players set {declcode} {self.mc_namespace}.cmcfc 0\n"
					out[func] += f"scoreboard players set {value_declcode} {self.mc_namespace}.cmcfc 0\n"

					if declarations[declcode] == DeclarationType["double"]:
						out[func] += f"execute store result score {declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {declcode} 1\n"
					elif declarations[declcode] == DeclarationType["float"]:
						out[func] += f"execute store result score {declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {declcode} {(10**FracCoef[DeclarationType['double']])/(10**FracCoef[DeclarationType['float']])}\n"
					else:
						out[func] += f"execute store result score {declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {declcode} {10**FracCoef[DeclarationType['double']]}\n"

					if declarations[value_declcode] == DeclarationType["double"]:
						out[func] += f"execute store result score {value_declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {value_declcode} 1\n"
					elif declarations[value_declcode] == DeclarationType["float"]:
						out[func] += f"execute store result score {value_declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {value_declcode} {(10**FracCoef[DeclarationType['double']])/(10**FracCoef[DeclarationType['float']])}\n"
					else:
						out[func] += f"execute store result score {value_declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {value_declcode} {10**FracCoef[DeclarationType['double']]}\n"


					if operand in [">", "<", ">=", "<="]:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int 1 run function {self.mc_namespace}:cmcfc/condition_operation "+"{"+f'objective:{self.mc_namespace}.cmcfc,left:{declcode},operand:"{operand}",right:{value_declcode}'+"}\n"
					elif operand == "==":
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int 1 run function {self.mc_namespace}:cmcfc/condition_operation " + "{" + f'objective:{self.mc_namespace}.cmcfc,left:{declcode},operand:"=",right:{value_declcode}' + "}\n"
					elif operand == "!=":
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int 1 run function {self.mc_namespace}:cmcfc/condition_operation " + "{" + f'objective:{self.mc_namespace}.cmcfc,left:{declcode},operand:"=",right:{value_declcode}' + "}\n"
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int 1 run function {self.mc_namespace}:cmcfc/not_operation " + "{" + f'objective:{self.mc_namespace}.cmcfc,var:{targetcode}' +"}\n"
					else:
						out[func] += f"scoreboard players operation {declcode} {self.mc_namespace}.cmcfc {operand}= {value_declcode} {self.mc_namespace}.cmcfc\n"
						if declarations[targetcode] == DeclarationType["double"]:
							out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int 1 run scoreboard players get {declcode} {self.mc_namespace}.cmcfc\n"
						elif declarations[targetcode] == DeclarationType["float"]:
							out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int {1/((10**FracCoef[DeclarationType['double']])/(10**FracCoef[DeclarationType['float']]))} run scoreboard players get {declcode} {self.mc_namespace}.cmcfc\n"
						else:
							out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {targetcode} int {1/(10**FracCoef[DeclarationType['double']])} run scoreboard players get {declcode} {self.mc_namespace}.cmcfc\n"
				elif operation == "CONDITION":
					condition_declcode = data[1]
					true_function_declcode = data[2]
					false_function_declcode = data[3]
					if false_function_declcode is None:
						false_function_declcode = "cmcfc/pass"

					out[func] += f"scoreboard players set {condition_declcode} {self.mc_namespace}.cmcfc 0\n"
					out[func] += f"execute store result score {condition_declcode} {self.mc_namespace}.cmcfc run data get storage {self.mc_namespace}:cmcfc {condition_declcode} 1\n"
					out[func] += f"function {self.mc_namespace}:cmcfc/condition " + "{" + f'objective:{self.mc_namespace}.cmcfc,condition:{condition_declcode},true_function:"{self.mc_namespace}:{true_function_declcode}",false_function:"{self.mc_namespace}:{false_function_declcode}"' + "}\n"
				elif operation == "CALL":
					out[func] += f"function {self.mc_namespace}:{data[1]}\n"
				elif operation == "RETURN":
					out[func] += f"return {data[1]}\n"
				elif operation == "ASM":
					text = data[1]
					asm_out = data[2]
					if asm_out is None:
						out[func] += f"{text}\n"
					else:
						out[func] += f"execute store result storage {self.mc_namespace}:cmcfc {asm_out} int 1 run {text}\n"
				elif operation == "ADDRESS":
					ptr_declcode = data[1]
					declcode = data[2]
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc {ptr_declcode} set value {declcode}i\n"
				elif operation == "DEREFERENCE":
					declcode = data[1]
					ptr_declcode = data[2]
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_ptr set from storage {self.mc_namespace}:cmcfc {ptr_declcode}\n"
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_val set value {declcode}i\n"
					out[func] += f"function {self.mc_namespace}:cmcfc/dereference_operation with storage {self.mc_namespace}:cmcfc\n"
				elif operation == "PTREQ":
					ptr_declcode = data[1]
					value_declcode = data[2]
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_ptr set from storage {self.mc_namespace}:cmcfc {ptr_declcode}\n"
					out[func] += f"data modify storage {self.mc_namespace}:cmcfc ptreq_val set value {value_declcode}i\n"
					out[func] += f"function {self.mc_namespace}:cmcfc/ptreq_operation with storage {self.mc_namespace}:cmcfc\n"
				else:
					raise CMCFCAssemblerError(f"Unknown operation: {operation}")
		if debug:
			mc_debug_out = ""
			for func in out:
				mc_debug_out += f"{func}:\n"
				mc_debug_out += f"{out[func]}\n"
			logger.debug(f"MCFUNCTION:\n{mc_debug_out}")
		return out



example_c_code = """
#include <stdio.h> 

int x;

void setup() {}
void loop() {
	if (x < 100) {
		x += 1;
	} else {
		printf("Hello World!");
		x = 0;
	}
}
"""

