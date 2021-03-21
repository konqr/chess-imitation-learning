# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 23:15:51 2021

@author: konqr
"""
import chess.pgn
import numpy as np

import chess.engine
import os
cwd = os.getcwd()
ENGINE = chess.engine.SimpleEngine.popen_uci(cwd+'/sf.exe')

chess_dict = {
                'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
                'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
                'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
                'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
                'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
                'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
                'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
                'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
                'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
                'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
                'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
                'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
                '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
            }

def make_matrix(board): 
    pgn = board.epd()
    foo = []  
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append('.')
            else:
                foo2.append(thing)
        foo.append(foo2)
    return foo

def translate(matrix,chess_dict):
    rows = []
    for row in matrix:
        terms = []
        for term in row:
            terms.append(chess_dict[term])
        rows.append(terms)
    return rows

def state_from_board(board):
    return translate(make_matrix(board),chess_dict)

class ChessEnv:
    
    def __init__(self,sf_depth=20, mate_reward=1e5):
        self.board = None
        self.pieces = ['','p','b','k','q','n','r','P','B','K','Q','N','R','P']
        self.columns= ['a','b','c','d','e','f','g','h']
        self.rows = range(1,9)
        self.sf_depth = sf_depth
        self.mate_reward = mate_reward
        
    def reset(self):
        self.board = chess.Board()
        return self
    
    def step(self,action):
        '''

        Parameters
        ----------
        action : Algebraic notation eg. Qa1

        Returns
        -------
        None.

        '''
        board_before = self.board
        self.board.push_uci(action)
        #reward =  None #just for completion's sake, we want to build a reward model later
        reward = self.reward()
        done = self.board.is_game_over()
        info = [board_before, self.action, self.board] 
        return state_from_board(self.board), reward, done, info
        
    def update(self,board):
        self.board = board
        return self
    
    def render(self):
        print("\n")
        print(self.board)
        print("\n")
        
    def action_space(self):
        action_space = []
        for i in self.columns:
            for j in self.rows:
                for m in self.columns:
                    for n in self.rows:
                        for p in self.pieces:
                            action_space.append(i+j+m+n+p)
        return action_space
    
    def reward(self):
        board = self.board
        info = ENGINE.analyse(board, chess.engine.Limit(depth=self.sf_depth))
        return info['score'].relative.score(mate_score=self.mate_reward)