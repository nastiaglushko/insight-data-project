def get_patternlist():#sentence (S)
	s="'ROOT'"

	#verb phrase (VP)
	vp="'VP > S|SINV|SQ'"
	vp_q="'MD|VBZ|VBP|VBD > (SQ !< VP)'"

	#clause (C)
	c="'S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]'"

	#T-unit (T)
	t="'S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]'"

	#dependent clause (DC)
	dc="'SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

	#complex T-unit (CT)
	ct="'S|SBARQ|SINV|SQ [> ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP]] << (SBAR < (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])]))'"

	#coordinate phrase (CP)
	cp="'ADJP|ADVP|NP|VP < CC'"

	#complex nominal (CN)
	cn1="'NP !> NP [<< JJ|POS|PP|S|VBG | << (NP $++ NP !$+ CC)]'"
	cn2="'SBAR [<# WHNP | <# (IN < That|that|For|for) | <, S] & [$+ VP | > VP]'"
	cn3="'S < (VP <# VBG|TO) $+ VP'"

	#fragment clause
	fc="'FRAG > ROOT !<< (S|SINV|SQ [> ROOT <, (VP <# VB) | <# MD|VBZ|VBP|VBD | < (VP [<# MD|VBP|VBZ|VBD | < CC < (VP <# MD|VBP|VBZ|VBD)])])'"

	#fragment T-unit
	ft="'FRAG > ROOT !<< (S|SBARQ|SINV|SQ > ROOT | [$-- S|SBARQ|SINV|SQ !>> SBAR|VP])'"

	#list of patterns to search for
	patternlist=[s,vp,c,t,dc,ct,cp,cn1,cn2,cn3,fc,ft,vp_q]

	return patternlist
