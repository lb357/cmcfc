import os
import json
from logger import *
import std
import shutil


class CMCFCDatapack(object):
	def __init__(self, target_namespace: str = None, path: str = None, cmcfc_path: str = None):
		self.DP_VERSION = 0
		self.PACK_FORMAT = 71

		if target_namespace is None:
			self.target_namespace = "cmcfc_default_namespace"
		else:
			self.target_namespace = target_namespace
		if path is None:
			self.path = os.getcwd()
		else:
			self.path = path

		if cmcfc_path is None:
			self.cmcfc_path = os.path.dirname(__file__)
		else:
			self.cmcfc_path = cmcfc_path

	def load_datapack(self, path: str) -> dict:
		if not os.path.exists(os.path.join(path, "pack.mcmeta")):
			logger.error("Datapack not found!")
		with open(os.path.join(path, "pack.mcmeta"), "r") as pack_file:
			pack_data = json.loads(pack_file.read())
		if std.CMCFC_VERSION != pack_data["cmcfc"]["cmcfc_version"]:
			logger.warn("This datapack was created by different CMCFC version!")
		if self.DP_VERSION != pack_data["cmcfc"]["dp_version"]:
			logger.warn("This datapack was created by different CMCFCDatapack (datapack env) version!")
		if self.PACK_FORMAT != pack_data["pack"]["pack_format"]:
			logger.warn("This datapack was created for different MC version!")
		self.path = path
		self.target_namespace = pack_data["cmcfc"]["target_namespace"]

		includes = {}

		for root, dirs, files in os.walk(os.path.join(os.path.abspath(self.path), "src")):
			for file in files:
				with open(os.path.join(root, file), "r") as include_file:
					includes[f'"{file}"'] = include_file.read()
		for file in std.stdlibs:
			includes[file] = std.stdlibs[file]
		return includes

	def create_datapack(self, name: str):
		pack_meta = {
			"pack": {
				"pack_format": 71,
				"description": f"Datapack {name} / Compiled using CMCFC",
			},
			"cmcfc": {
				"dp_version": self.DP_VERSION,
				"cmcfc_version": std.CMCFC_VERSION,
				"target_namespace": self.target_namespace
			}
		}

		path = os.path.join(self.path, name)
		if os.path.exists(path):
			logger.error("The datapack already exists!")
		else:
			self.path = path
			os.mkdir(self.path)
			os.mkdir(os.path.join(self.path, "data"))
			os.mkdir(os.path.join(self.path, "static"))
			os.mkdir(os.path.join(self.path, "src"))
			with open(os.path.join(self.path, "pack.mcmeta"), "w") as pack_file:
				pack_file.write(json.dumps(pack_meta, indent=4))
			with open(os.path.join(self.path, "src", "main.c"), "w") as main_file:
				main_file.write(std.default_main)

	def build_datapack(self, mcfunctions: dict):
		data_path = os.path.join(self.path, "data")
		cmcfc_std_path = os.path.join(self.cmcfc_path, "cmcfc_std")
		static_path = os.path.join(self.path, "static")
		mctagfunc_path = os.path.join(data_path, "minecraft", "tags", "function")

		if os.path.exists(data_path):
			shutil.rmtree(data_path)
		os.mkdir(data_path)
		os.mkdir(os.path.join(data_path, "minecraft"))
		os.mkdir(os.path.join(data_path, "minecraft", "tags"))
		os.mkdir(mctagfunc_path)
		# os.mkdir(os.path.join(data_path, self.target_namespace))
		# os.mkdir(os.path.join(data_path, self.target_namespace, "function"))
		shutil.copytree(cmcfc_std_path, os.path.join(data_path, self.target_namespace, "function", "cmcfc"),
						dirs_exist_ok=True)

		if os.path.exists(static_path):
			shutil.copytree(static_path, data_path, dirs_exist_ok=True)

		with open(os.path.join(mctagfunc_path, "load.json"), "w") as load_file:
			load_file.write(json.dumps({"values": [f"{self.target_namespace}" + ":setup"]}, indent=4))
		with open(os.path.join(mctagfunc_path, "tick.json"), "w") as tick_file:
			tick_file.write(json.dumps({"values": [f"{self.target_namespace}" + ":loop"]}, indent=4))
		open(os.path.join(data_path, self.target_namespace, "function", "setup.mcfunction"), "w").close()
		open(os.path.join(data_path, self.target_namespace, "function", "loop.mcfunction"), "w").close()

		for func in mcfunctions:
			with open(
					os.path.join(data_path,self.target_namespace, "function", f"{func}.mcfunction"), "w"
			) as function_file:
				function_file.write(mcfunctions[func])



if __name__ == "__main__":
	datapack = CMCFCDatapack(cmcfc_path=os.path.join(os.path.dirname(__file__), ".."))
	datapack.create_datapack("test_datapack")
	print(datapack.load_datapack(os.path.join(os.path.dirname(__file__), "test_datapack")))
	datapack.build_datapack({1:"say 1\n say2"})
