import pefile
import binascii
import os
from ruleset import *

class checker():

    def __init__(self, data):
        self.data = data
        self.nEntryPointSection = self.nEntryPointSection()

    def nEntryPointSection(self):
        result = []
        #for section in self.pe.sections: 
        #    if section.VirtualAddress >= self.getAddressOfEntryPoint():
        #        result.append(binascii.hexlify(section.get_data()).decode().upper())



        result.append(binascii.hexlify(self.data).upper().decode())
        return result

    def run(self):
        for EP in self.nEntryPointSection:
            for protector in ruleset.keys():
                for rules in ruleset[protector]:
                    rule, version = rules
                    if re.compile(rule).search(EP):
                        return version
