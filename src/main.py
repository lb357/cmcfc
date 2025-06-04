import argparse
import std
import os
from logger import logger
from cmcfc import CMCFCPreprocessor, CMCFCCompiler, CMCFCAssembler
from datapack import CMCFCDatapack


def create_dir_path(string):
	path, namespace = string.split("@")
	if path == "" and namespace != "":
		return None, namespace
	elif os.path.isdir(path) and namespace != "":
		return path, namespace
	else:
		logger.error(f"{path} not correct path@namespace!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog='CMCFC',
		description=f'CMCFC (v.{std.CMCFC_VERSION}) is compiler/translator from C89 to MCFunction (Minecraft {std.MINECRAFT_VERSION})',
		usage="%(prog)s [options] (--help for more info)",
		epilog="MIT License / 2025 Leonid Briskindov",
	)

	parser.add_argument("-create", type=create_dir_path, help="Create datapack (path@namespace)")
	parser.add_argument("-build", help="Build datapack (path)")

	parser.add_argument("-std_path", type=str, help="PATH TO CMCFC STD FILES")
	parser.add_argument("-debug", type=bool, help="DEBUG MODE")
	args = parser.parse_args()
	if args.debug:
		logger.info(f"ARGS:\n{args}")
	if args.create is not None and args.build is not None:
		logger.error("There are too many arguments")
	elif args.build is not None:
		datapack = CMCFCDatapack(cmcfc_path=args.std_path)
		includes = datapack.load_datapack(args.build)

		preprocessor = CMCFCPreprocessor()
		compiler = CMCFCCompiler(mc_namespace=datapack.target_namespace)
		assembler = CMCFCAssembler(mc_namespace=datapack.target_namespace)

		c_code = includes['"main.c"']
		preprocessed_c_code = preprocessor.preprocess(c_code, includes, debug=args.debug)
		asm_code, declarations, process_data = compiler.compile(preprocessed_c_code, debug=args.debug)
		mcf_code = assembler.assemble(asm_code, declarations, process_data, debug=args.debug)

		datapack.build_datapack(mcf_code)
	elif args.create is not None:
		datapack = CMCFCDatapack(path=args.create[0], target_namespace=args.create[1], cmcfc_path=args.std_path)
		datapack.create_datapack(args.create[1])
	else:
		logger.error("Unknown command! (See --help)")