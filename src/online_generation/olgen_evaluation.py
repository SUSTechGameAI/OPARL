# """
#   @Time : 2022/2/14 20:24
#   @Author : Ziqi Wang
#   @File : olgen_evaluation.py
# """
#
# from abc import abstractmethod
#
#
# class Evaluator:
#     def __init__(self, *terms):
#         self.buffers = []
#         self.terms = terms
#
#     def update(self, **kwargs):
#         for term in self.terms:
#             term.update(**kwargs)
#             pass
#         pass
#
#     def take_out(self):
#         pass
#
#
# class EvaluationTerm:
#     pass
#
#
# class FunEvaluationTerm:
#     pass
#
#
# class PlayabilityEvaluationTerm:
#     pass
#
#
# class ControllabilityEvaluationTerm:
#     pass
#
#
# class MDFC_ErrorEvaluationTerm:
#     pass
#
#
# class DDA_ErrorEvaluationTerm:
#     pass
