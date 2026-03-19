# experimental setup
config_DLG = dict(signed=False,
              boxed=False,
              lr=1, 
              optim='adam',
              max_iterations=300,
              total_variation=1e-6,
              init='randn',
              lr_decay=False,
              method = 'DLG'
              )

config_iDLG = dict(signed=False, 
              boxed=False, 
              lr=1, 
              optim='adam', 
              max_iterations=300, 
              total_variation=1e-6,
              init='randn',
              lr_decay=False, 
              method = 'iDLG'
              ) 

config_IG = dict(signed=True,
              boxed=False, 
              cost_fn='sim', 
              indices='def', 
              weights='equal',
              lr=0.1,
              optim='adam', 
              restarts=1,
              max_iterations=3200,
              total_variation=1e-6, 
              init='randn', 
              filter='none', 
              lr_decay=True,
              scoring_choice='loss',
              method = 'IG'
              )

config_ISG = dict(signed=True,
              boxed=False,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=3200, 
              total_variation=1e-6, 
              kl_value = 1e-8,
              init='randn',
              filter='none',
              lr_decay=True, 
              scoring_choice='loss',
              method='ISG'
              )

config_NAS = dict(signed=True,
              boxed=False,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.001, #学习率
              optim='adam',
              restarts=1,
              max_iterations = 100, #迭代次数
              total_variation = 1e-6, #tv正则化系数
              kl_value = 1e-8,
              init='randn',
              filter='none',
              lr_decay=True, #是否设置学习率衰减
              scoring_choice='loss',
              method='GINAS'
              ) 