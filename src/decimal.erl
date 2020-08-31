%% @doc An Erlang decimal arithmetic library.


-module(decimal).

-type decimal() :: {Base :: integer(), Exp :: integer()}.
-type round_type() :: down | ceil | floor | half_up | half_down.
-type binary_opts() :: #{ pretty := boolean() }.

-export_type([
              decimal/0,
              round_type/0,
              binary_opts/0
             ]).

%% ====================================================================
%% API functions
%% ====================================================================
-export([
         neg/1,
         abs/1,
         add/2,
         sub/2,
         mul/2,
         divide/3,
         cmp/2,
         normalize/1,
         round/3
        ]).

-export([
         from_binary/1,
         from_list/1,
         from_number/1,
         to_binary/1,
         to_binary/2,
         to_number/1
        ]).

-spec neg(decimal()) -> decimal().
neg({Base, Exp}) -> {-Base, Exp}.

-spec abs(decimal()) -> decimal().
abs({Base, Exp}) -> {erlang:abs(Base), Exp}.

-spec add(decimal(), decimal()) -> decimal().
add(Dec1, {0, _}) -> Dec1;
add({0, _}, Dec2) -> Dec2;
add({Base1, Exp}, {Base2, Exp}) ->
    {Base1 + Base2, Exp};
add({Base1, Exp1}, {Base2, Exp2}) when Exp1 < Exp2 ->
    {Base1 + Base2 * pow10(Exp2 - Exp1), Exp1};
add({Base1, Exp1}, {Base2, Exp2}) -> % when Exp1 > Exp2 ->
    {Base1 * pow10(Exp1 - Exp2) + Base2, Exp2}.


-spec sub(decimal(), decimal()) -> decimal().
sub(Dec1, Dec2) ->
    add(Dec1, neg(Dec2)).


-spec mul(decimal(), decimal()) -> decimal().
mul({Base1, Exp1}, {Base2, Exp2}) ->
    {Base1 * Base2, Exp1 + Exp2}.


-spec divide(decimal(), decimal(), integer()) -> decimal().
divide(_Dec1, {0, _Exp2}, _Precision) -> error(badarith);
divide({Base1, Exp1}, {Base2, Exp2}, Precision) ->
    ExpMin = min(Exp1, Exp2),
    Base = (Base1 * pow10(Exp1 - ExpMin + Precision)) div (Base2 * pow10(Exp2 - ExpMin)),
    {Base, -Precision}.


-spec cmp(decimal(), decimal()) -> -1 | 0 | 1.
cmp({0, _Exp1}, {0, _Exp2}) -> 0;
cmp({Base1, _Exp1}, {Base2, _Exp2}) when Base1 >= 0, Base2 =< 0 -> 1;
cmp({Base1, _Exp1}, {Base2, _Exp2}) when Base1 =< 0, Base2 >= 0 -> -1;
cmp({Base1, Exp}, {Base2, Exp}) ->
    icmp(Base1, Base2);
cmp({Base1, Exp1}, {Base2, Exp2}) when Exp1 < Exp2 ->
    icmp(Base1, Base2 * pow10(Exp2 - Exp1));
cmp({Base1, Exp1}, {Base2, Exp2}) -> % when Exp1 > Exp2 ->
    icmp(Base1 * pow10(Exp1 - Exp2), Base2).


-spec normalize(decimal()) -> decimal().
normalize({0, _Exp}) -> {0, 0};
normalize({Base, Exp}) ->
    normalize_(Base, Exp).


-spec round(round_type(), decimal(), Precision :: integer()) -> decimal().
round(Type, {_Base, Exp} = Dec, Precision) ->
    if -Precision - Exp > 0 ->
           round_(Type, Dec, Precision);
       true ->
           Dec
    end.


round_(down, {Base, Exp}, Precision) ->
    Delta = -Precision - Exp,
    {Base div pow10(Delta), -Precision};

round_(ceil, {Base, Exp}, Precision) ->
    Delta = -Precision - Exp,
    P = pow10(Delta),
    Base0 = Base div P,
    Diff = Base - Base0 * P,
    Base1 = if Diff > 0 -> Base0 + 1;
               true -> Base0
            end,
    {Base1, -Precision};

round_(floor, {Base, Exp}, Precision) ->
    Delta = -Precision - Exp,
    P = pow10(Delta),
    Base0 = Base div P,
    Diff = Base - Base0 * P,
    Base1 = if Diff < 0 -> Base0 - 1;
               true -> Base0
            end,
    {Base1, -Precision};

round_(half_up, {Base, Exp}, Precision) ->
    Delta = -Precision - Exp,
    P = pow10(Delta - 1),
    Data = Base div P,
    Base0 = Data div 10,
    LastDigit = erlang:abs(Data - Base0 * 10),
    Base1 = if LastDigit >= 5, Data > 0 ->
                   Base0 + 1;
               LastDigit >= 5, Data < 0 ->
                   Base0 - 1;
               true ->
                   Base0
            end,
    {Base1, -Precision};

round_(half_down, {Base, Exp}, Precision) ->
    Delta = -Precision - Exp,
    P = pow10(Delta - 1),
    Data = Base div P,
    Base0 = Data div 10,
    LastDigit = erlang:abs(Data - Base0 * 10),
    Base1 = if LastDigit > 5, Data > 0 ->
                   Base0 + 1;
               LastDigit > 5, Data < 0 ->
                   Base0 - 1;
               true ->
                   Base0
            end,
    {Base1, -Precision}.


-spec from_binary(binary()) -> decimal().
from_binary(Bin) ->
    parse_base(Bin, <<>>).

parse_base(<<$-, Rest/binary>>, <<>>) ->
    parse_base(Rest, <<$->>);
parse_base(<<$., Rest/binary>>, Acc) ->
    parse_fraction(Rest, Acc, 0);
parse_base(<<X, Rest/binary>>, Acc) when X >= $0, X =< $9 ->
    parse_base(Rest, <<Acc/binary, X>>);
parse_base(<<X, Rest/binary>>, Acc) when X =:= $E; X =:= $e ->
    parse_exp(Rest, Acc, 0, <<>>);
parse_base(<<>>, Acc) ->
    {binary_to_integer(Acc),0};
parse_base(_, _) ->
    error(badarg).

parse_fraction(<<X, Rest/binary>>, Acc, E) when X >= $0, X =< $9 ->
    parse_fraction(Rest, <<Acc/binary, X >>, E-1);
parse_fraction(<<X, Rest/binary>>, Acc, E) when X =:= $E; X =:= $e ->
    parse_exp(Rest, Acc, E, <<>>);
parse_fraction(<<>>, Acc, E) ->
    {binary_to_integer(Acc), E};
parse_fraction(_, _, _) ->
    error(badarg).

parse_exp(<<$-, Rest/binary>>, Base, E, <<>>) ->
    parse_exp(Rest, Base, E, <<$->>);
parse_exp(<<$+, Rest/binary>>, Base, E, <<>>) ->
    parse_exp(Rest, Base, E, <<>>);
parse_exp(<<X, Rest/binary>>, Base, E, Acc) when X >= $0, X =< $9 ->
    parse_exp(Rest, Base, E, <<Acc/binary, X>>);
parse_exp(<<>>, Base, E, Acc) ->
    {binary_to_integer(Base), E + binary_to_integer(Acc)};
parse_exp(_, _, _, _) ->
    error(badarg).

-spec from_list(iodata()) -> decimal().
from_list(List) when is_list(List) ->
    from_binary(iolist_to_binary(List)).

from_number(Int) when is_integer(Int) -> {Int, 0};
from_number(0.0) -> {0, 0};
from_number(Float) when is_float(Float) ->
    {Frac, Exp} = mantissa_exponent(Float),
    {Place, Digits} = from_float_(Float, Exp, Frac),
    Decimal = {B, E} = to_decimal(Place, [$0 + D || D <- Digits]),
    case Float < 0.0 of
        true -> {-B, E};
        false -> Decimal
    end.


-spec to_binary(decimal()) -> binary().
to_binary(Dec) ->
    to_binary(Dec, #{pretty => false}).

-spec to_binary(decimal:decimal(), Opts) -> binary() when
      Opts :: binary_opts().
to_binary({0, _}, _Opts) ->
    <<"0.0">>;
to_binary({Int, 0}, _Opts) ->
    <<(integer_to_binary(Int))/binary, ".0">>;
to_binary({Base, E}, #{pretty := Pretty}) ->
    Sign =
        case Base < 0 of
            true -> <<$->>;
            false -> <<>>
        end,
    Bin = integer_to_binary(erlang:abs(Base)),
    Size = byte_size(Bin),
    case Size + E - 1 of
        AE when E < 0 andalso ((not Pretty) orelse (AE > -6)) ->
            case AE < 0 of
                true ->
                    <<Sign/binary, "0.", (binary:copy(<<$0>>, -(AE + 1)))/binary, Bin/binary>>;
                false ->
                    Shift = AE + 1,
                    <<B:Shift/binary, R/binary>> = Bin,
                    <<Sign/binary, B/binary, $., R/binary>>
            end;
        AE when E >= 0 andalso ((not Pretty) orelse (AE < 6)) ->
            <<Sign/binary, Bin/binary,(binary:copy(<<$0>>, E))/binary, ".0">>;
        AE when Size =:= 1->
            <<Sign/binary, Bin/binary, ".0", (e(AE))/binary>>;
        AE ->
            <<B:1/binary, R/binary>> = Bin,
            <<Sign/binary, B/binary, $., R/binary, (e(AE))/binary>>
    end.


e(0) -> <<>>;
e(E) -> <<$e, (integer_to_binary(E))/binary>>.



-spec to_number(decimal()) -> number().
to_number({Base, Exp}) ->
    if Exp =:= 0 -> Base;
       Exp > 0 -> Base * pow10(Exp);
       true -> Base / pow10(-Exp)
    end.

%% ====================================================================
%% Internal functions
%% ====================================================================

-spec normalize_(integer(), integer()) -> {integer(), integer()}.
normalize_(Base, Exp) ->
    if Base rem 10 =:= 0 ->
           normalize_(Base div 10, Exp + 1);
       true ->
           {Base, Exp}
    end.


-spec icmp(integer(), integer()) -> -1 | 0 | 1. 
icmp(Int1, Int2) -> 
    if Int1 =:= Int2 -> 0;
       Int1 < Int2 -> -1;
       true -> 1  % Int1 > Int2
    end.


-spec pow10(non_neg_integer()) -> non_neg_integer().
pow10(Pow) when is_integer(Pow), Pow >= 0 ->
    ipow(10, Pow).


-spec ipow(N :: integer(), Pow :: non_neg_integer()) -> non_neg_integer().
ipow(0, 0) -> error(badarith);
ipow(_N, 0) -> 1;
ipow(0, _Pow) -> 0;
ipow(N, 1) -> N;
ipow(N, 2) -> N * N;
ipow(N, 3) -> N * N * N;
ipow(_N, Pow) when Pow < 0 -> error(badarith);
ipow(N, Pow) ->
    ipow(N, Pow, 1).


-spec ipow(N :: integer(), Pow :: non_neg_integer(), Acc :: integer()) -> integer().
ipow(N, 1, Acc) -> N * Acc;
ipow(N, Pow, Acc) ->
    NewAcc = if (Pow band 1) =/= 0 -> N * Acc;
                true -> Acc
             end,
    ipow(N * N, Pow bsr 1, NewAcc).


-define(BIG_POW, (1 bsl 52)).
-define(MIN_EXP, (-1074)).
mantissa_exponent(F) ->
    case <<F:64/float>> of
        <<_S:1, 0:11, M:52>> -> % denormalized
            E = log2floor(M),
            {M bsl (53 - E), E - 52 - 1075};
        <<_S:1, BE:11, M:52>> when BE < 2047 ->
            {M + ?BIG_POW, BE - 1075}
    end.


from_float_(Float, Exp, Frac) ->
    Round = (Frac band 1) =:= 0,
    if
        Exp >= 0 ->
            BExp = 1 bsl Exp,
            if
                Frac =:= ?BIG_POW ->
                    scale(Frac * BExp * 4, 4, BExp * 2, BExp, Round, Round, Float);
                true ->
                    scale(Frac * BExp * 2, 2, BExp, BExp, Round, Round, Float)
            end;
        Exp < ?MIN_EXP ->
            BExp = 1 bsl (?MIN_EXP - Exp),
            scale(Frac * 2, 1 bsl (1 - Exp), BExp, BExp,
                  Round, Round, Float);
        Exp > ?MIN_EXP, Frac =:= ?BIG_POW ->
            scale(Frac * 4, 1 bsl (2 - Exp), 2, 1,
                  Round, Round, Float);
        true ->
            scale(Frac * 2, 1 bsl (1 - Exp), 1, 1,
                  Round, Round, Float)
    end.

scale(R, S, MPlus, MMinus, LowOk, HighOk, Float) ->
    Est = int_ceil(math:log10(erlang:abs(Float)) - 1.0e-10),
    %% Note that the scheme implementation uses a 326 element look-up
    %% table for int_pow(10, N) where we do not.
    if
        Est >= 0 ->
            fixup(R, S * pow10(Est), MPlus, MMinus, Est, LowOk, HighOk);
        true ->
            Scale = pow10(-Est),
            fixup(R * Scale, S, MPlus * Scale, MMinus * Scale, Est, LowOk, HighOk)
    end.


fixup(R, S, MPlus, MMinus, K, LowOk, HighOk) ->
    TooLow = if
                 HighOk -> R + MPlus >= S;
                 true -> R + MPlus > S
             end,
    case TooLow of
        true ->
            {K + 1, generate(R, S, MPlus, MMinus, LowOk, HighOk)};
        false ->
            {K, generate(R * 10, S, MPlus * 10, MMinus * 10, LowOk, HighOk)}
    end.


generate(R0, S, MPlus, MMinus, LowOk, HighOk) ->
    D = R0 div S,
    R = R0 rem S,
    TC1 = if
              LowOk -> R =< MMinus;
              true -> R < MMinus
          end,
    TC2 = if
              HighOk -> R + MPlus >= S;
              true -> R + MPlus > S
          end,
    case {TC1, TC2} of
        {false, false} ->
            [D | generate(R * 10, S, MPlus * 10, MMinus * 10, LowOk, HighOk)];
        {false, true} ->
            [D + 1];
        {true, false} ->
            [D];
        {true, true} when R * 2 < S ->
            [D];
        {true, true} ->
            [D + 1]
    end.


to_decimal(Place, S) ->
    {list_to_integer(S), Place - length(S)}.


int_ceil(X) when is_float(X) ->
    T = trunc(X),
    case (X - T) of
        Neg when Neg < 0 -> T;
        Pos when Pos > 0 -> T + 1;
        _ -> T
    end.


log2floor(Int) when is_integer(Int), Int > 0 ->
    log2floor(Int, 0).
log2floor(0, N) ->
    N;
log2floor(Int, N) ->
    log2floor(Int bsr 1, 1 + N).

