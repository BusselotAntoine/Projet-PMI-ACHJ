
clear; close all; clc;
%% --- PARAMÈTRES PHYSIQUES ET GÉOMÉTRIQUES ---
R = 0.01;       % Rayon du disque (1 cm)
D = 0.01;       % Distance entre les disques (1 cm)
epsilon = D/R;  % Paramètre sans dimension (rapport de forme)

a = 0; % Borne inférieure du domaine normalisé
b = 1; % Borne supérieure du domaine normalisé

% Noyau de l'équation intégrale de Love (modélise l'interaction électrostatique)
% K(x,y) est symétrique et dépend de la distance relative entre les points
K = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));

%% --- CONFIGURATION DES TESTS DE CONVERGENCE ---
n_values = [10, 20, 40, 80];        % Différentes discrétisations à tester
errors_classic = zeros(size(n_values)); % Stockage erreurs (méthode intégrale native)
errors_gauss   = zeros(size(n_values)); % Stockage erreurs (méthode quadrature Gauss)

%% --- RÉSOLUTIONS LINÉAIRES ---
% =========================================================================
% FONCTION : solve_linear_classic
% ROLE : Résout l'équation intégrale de Love par une méthode de Galerkin
%        avec des fonctions de base linéaires (P1) et utilise l'intégrateur
%        adaptatif natif de MATLAB pour remplir la matrice.
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Nombre d'intervalles de discrétisation.
%   K   : (Handle) Fonction du noyau K(x,y).
%   a,b : (Scalaire) Bornes du domaine d'intégration.
% SORTIES :
%   x   : (Vecteur) Points de la grille de discrétisation.
%   u   : (Vecteur) Solution (densité de charge) aux points x.
% =========================================================================
function [x, u] = solve_linear_classic(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    for i = 1:n+1
        for j = 1:n+1
            % Base de fonctions "chapeau" (P1 linéaire par morceaux)
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);
            % Calcul de l'élément de matrice par intégration adaptive
            A(i,j) = integral(@(y) K(x(i), y).*phi_j(y), y_min, y_max);
        end
    end
    % Résolution du système linéaire (I - A)u = 1
    u = (eye(n+1) - A) \ ones(n+1,1);
end

% =========================================================================
% FONCTION : gauss_legendre
% ROLE : Calcule les nœuds et les poids de la quadrature de Gauss-Legendre
%        sur l'intervalle standard [-1, 1] en utilisant l'algorithme de 
%        Golub-Welsch (diagonalisation de la matrice de Jacobi).
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Ordre de la quadrature (nombre de points).
% SORTIES :
%   x   : (Vecteur) Nœuds de quadrature sur [-1, 1].
%   w   : (Vecteur) Poids de quadrature associés.
% =========================================================================
function [x,w] = gauss_legendre(n)
    beta = 0.5 ./ sqrt(1 - (2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1); % Matrice de Jacobi
    [V,D] = eig(T);
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * (V(1,idx)').^2; % Poids de quadrature
end

% =========================================================================
% FONCTION : solve_linear_gauss
% ROLE : Similaire à solve_linear_classic, mais remplace l'intégrateur 
%        natif par une quadrature de Gauss-Legendre de degré fixe.
%        C'est beaucoup plus performant pour les calculs itératifs.
% -------------------------------------------------------------------------
% ENTRES :
%   n   : (Scalaire) Nombre d'intervalles.
%   K   : (Handle) Le noyau de l'équation.
%   a,b : (Scalaire) Bornes du domaine.
% SORTIES :
%   x   : (Vecteur) Grille de points.
%   u   : (Vecteur) Solution approchée.
% =========================================================================
function [x, u] = solve_linear_gauss(n, K, a, b)
    h = (b - a)/n;
    x = linspace(a, b, n+1);
    A = zeros(n+1);

    quad_order = 20; % Ordre de la quadrature pour chaque segment
    [gl_x, gl_w] = gauss_legendre(quad_order);

    for i = 1:n+1
        xi = x(i);
        for j = 1:n+1
            phi_j = @(y) max(0, 1 - abs(y - x(j))/h);
            y_min = max(a, x(j) - h);
            y_max = min(b, x(j) + h);

            if y_max <= y_min, A(i,j) = 0; continue; end

            % Changement de variable vers l'intervalle [y_min, y_max]
            yq = (y_max + y_min)/2 + (y_max - y_min)/2 * gl_x;
            wq = gl_w * (y_max - y_min)/2;

            A(i,j) = sum( K(xi, yq).*phi_j(yq).*wq );
        end
    end
    u = (eye(n+1) - A) \ ones(n+1,1);
end

%% --- CALCUL DE L'ERREUR ET RÉFÉRENCE ---

n_ref = 800; % Solution très fine pour servir de "vérité terrain"
[x_ref, u_ref] = solve_linear_gauss(n_ref, K, a, b);

for k = 1:length(n_values)
    n = n_values(k);
    
    % Test méthode classique
    [x_c, u_c] = solve_linear_classic(n, K, a, b);
    u_ref_interp = interp1(x_ref, u_ref, x_c);
    errors_classic(k) = max(abs(u_c - u_ref_interp'));

    % Test méthode Gauss
    [x_g, u_g] = solve_linear_gauss(n, K, a, b);
    u_ref_interp2 = interp1(x_ref, u_ref, x_g);
    errors_gauss(k) = max(abs(u_g - u_ref_interp2'));
end


% Valeurs de n demandées par l'énoncé

errors = zeros(size(n_values));
h_values = 1 ./ n_values;

% Calcule de la solution de référence
n_ref = 200; %Beaucoup de points pour la référence
% Nouveau code (inversion de u et x + ajout des bornes 0, 1)
[x_ref, u_ref] = solve_linear_classic(n_ref, K, 0, 1);

% --- calcul de l'erreur ---
for k = 1:length(n_values)
    n = n_values(k);
    % Nouveau code
    [x_n, u_n] = solve_linear_classic(n, K, 0, 1);

    % Interpolation de u_ref sur la grille plus grossière x_n pour comparer
    u_ref_interp = interp1(x_ref, u_ref, x_n);

    % Calcul de l'erreur
    errors(k) = max(abs(u_n - u_ref_interp'));
end


%% --- APPROCHE PAR INTERPOLATION GLOBALE (LAGRANGE) ---

% =========================================================================
% FONCTION : lagrange_basis
% ROLE : Calcule la valeur du j-ième polynôme de base de Lagrange 
%        passant par les nœuds 'x_nodes' évalué au(x) point(s) 'y'.
% -------------------------------------------------------------------------
% ENTRES :
%   j       : (Entier) Indice du nœud de base (1 à n).
%   x_nodes : (Vecteur) Points d'interpolation (nœuds).
%   y       : (Vecteur/Scalaire) Point(s) où évaluer le polynôme.
% SORTIES :
%   Lj      : (Vecteur/Scalaire) Valeur(s) de phi_j(y).
% =========================================================================
function Lj = lagrange_basis(j, x_nodes, y)
    n = length(x_nodes);
    Lj = ones(size(y));
    xj = x_nodes(j);
    for k = 1:n
        if k ~= j
            Lj = Lj .* ( (y - x_nodes(k)) / (xj - x_nodes(k)) );
        end
    end
end

% =========================================================================
% FONCTION : build_A_matrix
% ROLE : Construit la matrice de l'opérateur intégral en utilisant une
%        approximation par interpolation globale (Lagrange) au lieu 
%        d'une approximation locale par morceaux.
% -------------------------------------------------------------------------
% ENTRES :
%   x_nodes    : (Vecteur) Points d'interpolation choisis.
%   kernel     : (Handle) Fonction noyau K(x,y).
%   a,b        : (Scalaire) Intervalle d'intégration.
%   quad_order : (Entier) Nombre de points pour la quadrature numérique.
% SORTIES :
%   A          : (Matrice n x n) Matrice de Fredholm discrétisée.
% =========================================================================
function A = build_A_matrix(x_nodes, kernel, a, b, quad_order)
    n = length(x_nodes);
    A = zeros(n);
    [yq, wq] = gauss_legendre(quad_order);
    yq = 0.5*(b-a)*yq + 0.5*(b+a);
    wq = 0.5*(b-a)*wq;

    for i = 1:n
        xi = x_nodes(i);
        for j = 1:n
            Lj = lagrange_basis(j, x_nodes, yq);
            A(i,j) = sum((kernel(xi, yq) .* Lj) .* wq);
        end
    end
end

%% --- ANALYSE DES RÉSULTATS ET AFFICHAGE ---

% Question 5
fprintf('Q5\n');

n2 = 40; % Nombre de points
x2 = linspace(a, b, n2); % Points
A2 = build_A_matrix(x2,K,a,b,n2);
I2 = eye(n2);
d2 = ones(n2, 1); % Terme source d(x) = 1

u2 = (I2 - A2) \ d2;

figure(1);
plot(x_c, u_c);
title('Densité de charge équivalente u(x)');
xlabel('Position normalisée x');
ylabel('u(x)');


figure(2);
plot(x_g, u_g, 'LineWidth', 2);
title('Méthode linéaire avec quadrature de Gauss-Legendre');
xlabel('x');
ylabel('u(x)');
grid on;

figure(3);
plot(x2, u2);
title('Densité de charge équivalente u2(x)');
xlabel('Position normalisée x2');
ylabel('u2(x)');

fprintf('Regarder figure 1,2 & 3\n');

fprintf('\n');
% Question 7
fprintf('Q7\n');


% Graphique Erreur vs n 

figure(4);
subplot(1,2,1);
loglog(n_values, errors, 'r-o', 'LineWidth', 2);
grid on;
xlabel('Nombre de points n');
ylabel('Erreur e_n');
title('Erreur vs n');

% Graphique Erreur vs h 
subplot(1,2,2);
loglog(h_values, errors, 'b-s', 'LineWidth', 2);
title('Erreur vs h');
xlabel('Pas d''espace h');
ylabel('Erreur e_n');
hold on;
grid on;

% --- Ajout : droite de référence en h^2 ---
ref = errors(1) * (h_values / h_values(1)).^2;
loglog(h_values, ref, 'r--', 'LineWidth', 1.5);
legend('Erreur', 'Référence h^2');

% --- Ajout : calcul de l'ordre ---
p = polyfit(log(h_values), log(errors), 1);
disp(['Ordre de convergence estimé : p = ', num2str(p(1))]);

% Question 8 ----------
fprintf('Q8\n');

figure(5);
loglog(n_values, errors_classic, 'r-o', 'LineWidth', 2); hold on;
loglog(n_values, errors_gauss, 'b-s', 'LineWidth', 2);
grid on;
xlabel('n');
ylabel('Erreur max ||u - u_n||_\infty');
title('Comparaison des méthodes linéaires');
legend('Classique (integral)', 'Gauss-Legendre');
fprintf('Regarder figure 4 & 5\n');


fprintf('\n');

% Question 9 ----------
fprintf('Q9\n');
epsilon0 = 8.854e-12; 
R_val = 0.01;         
D_val = 0.01;         
epsilon_geom = D_val / R_val; 

K1 = @(x,y)(1/pi) * (1./(epsilon_geom^2 + (x-y).^2) + 1./(epsilon_geom^2 + (x+y).^2));

n_ref = 200; 
[x_res, u_res] = solve_linear_gauss(n_ref, K1, 0, 1);

valeur_integrale = trapz(x_res, u_res); 
C_numerique = 4 * epsilon0 * R_val * valeur_integrale;

fprintf('Résultat de l''intégrale numérique : %.4f\n', valeur_integrale);
fprintf('Capacité calculée (Love) : %.4e F\n', C_numerique);

fprintf('\n');

% Q10
fprintf('Q10\n');
     
S = pi * R_val^2;     

C_classique = (epsilon0 * S) / D_val;

fprintf('Capacité classique (S/D) : %.4e F\n', C_classique);
ecart_relatif = abs(C_numerique - C_classique) / C_numerique * 100;
fprintf('Écart relatif : %.2f %%\n', ecart_relatif);

D_values_eff = [0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001]; 
D_values_tot = [0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002 0.001];

fprintf('\n');

% Q11
fprintf('Q11\n');
fprintf('Regarder figure 7 , 9 & 10\n');
 

for D = D_values_eff
    epsilon = D/R_val;
    K_new = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));
    
    n_pts = 100;
    [x_u, u_sol] = solve_linear_gauss(n_pts, K_new, 0, 1);
    
    % Calculs des capacités
    C_Love = 4 * epsilon0 * R * trapz(x_u, u_sol);
    C_class = (epsilon0 * S) / D;

    figure(7); hold on;
    plot(x_u, u_sol, 'DisplayName', ['D = ' num2str(D*1000) ' mm']);
end


title('Influence de la distance sur la densité de charge u(x) pour la resolution gaussienne');
xlabel('x (normalisé)'); ylabel('u(x)');
legend; grid on; 



for D = D_values_eff
    epsilon = D/R_val;
    K_new = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));
    
    n_pts = 100;
    [x_u, u_sol] = solve_linear_classic(n_pts, K_new, 0, 1);
    
    % Calculs des capacités
    C_Love = 4 * epsilon0 * R * trapz(x_u, u_sol);
    C_class = (epsilon0 * S) / D;
    

    % Tracer u(x) pour comparer les profils
    figure(9); hold on;
    plot(x_u, u_sol, 'DisplayName', ['D = ' num2str(D*1000) ' mm']);
end

title('Influence de la distance sur la densité de charge u(x) pour la resolution lineaire');
xlabel('x (normalisé)'); ylabel('u(x)');
legend; grid on; 

for D = D_values_eff
    epsilon = D/R_val;
    K_new = @(x,y)(1/pi) * (1./(epsilon^2 + (x-y).^2) + 1./(epsilon^2 + (x+y).^2));
    
    n3 = 30; % Nombre de points
    x3 = linspace(a, b, n3); % Points
    A3 = build_A_matrix(x3,K_new,a,b,n3);
    I3 = eye(n3);
    d3 = ones(n3, 1); % Terme source d(x) = 1

    u_new = (I3 - A3) \ d3;
    
    % Calculs des capacités
    C_Love = 4 * epsilon0 * R * trapz(x3, u_new);
    C_class = (epsilon0 * S) / D;
    

    % Tracer u(x) pour comparer les profils
    figure(10); hold on;
    plot(x3, u_new, 'DisplayName', ['D = ' num2str(D*1000) ' mm']);
end

title('Influence de la distance sur la densité de charge u(x) pour la resolution avec polynôme de Lagrange globaux');
xlabel('x (normalisé)'); ylabel('u(x)');
legend; grid on; 

fprintf('\n');

%% ============================
%  MÉTHODE D'ADOMIAN 
% =============================

%% --- Paramètres numériques ---
N_ad = 100;             
x_ad = linspace(a,b,N_ad);  
quad_order = 40;      % Quadrature Gauss-Legendre
[uq, wq] = gauss_legendre(quad_order);

% Transformation vers [0,1]
yq = 0.5*(b-a)*uq + 0.5*(b+a);
wq = 0.5*(b-a)*wq;

%% --- Décomposition d'Adomian ---
max_iter = 40;        % Nombre de termes de la série
U_terms = zeros(max_iter, N_ad);

% Terme initial : u0(x) = 1
U_terms(1,:) = ones(1,N_ad);

for n = 1:max_iter-1
    u_prev = U_terms(n,:);
    u_next = zeros(1,N_ad);

    for i = 1:N_ad
        xi = x_ad(i);
        u_next(i) = sum( K(xi, yq) .* interp1(x_ad, u_prev, yq, 'linear') .* wq );
    end

    U_terms(n+1,:) = u_next;
end

u_adomian = sum(U_terms,1);

figure(11);
plot(x_ad, u_adomian, 'LineWidth', 2);
xlabel('x'); ylabel('u(x)');
title('Méthode d''Adomian appliquée à l''équation de Love');
grid on;

errors3 = zeros(max_iter,1);

for k = 1:max_iter
    u_adomian2 = sum(U_terms(1:k,:),1);
    u_ref_interp = interp1(x_ref, u_ref, x_ad);
    errors3(k) = max(abs(u_adomian2 - u_ref_interp));
end

figure(12);
semilogy(1:max_iter, errors3, 'b-o', 'LineWidth', 2);
xlabel('Nombre de termes N');
ylabel('Erreur ||u^{(N)} - u_{ref}||_\infty');
title('Convergence lente de la méthode d''Adomian');
grid on;

%% ============================
%  MÉTHODE COLLOCATION
% =============================

% =========================================================================
% FONCTION : cheb4_poly
% ROLE : Calcule tous les polynômes de Tchebychev
%        W_0(x), W_1(x), ..., W_N(x) en un point donné x.
% -------------------------------------------------------------------------
% DESCRIPTION :
%   Cette fonction renvoie un vecteur ligne contenant les valeurs des
%   polynômes W_i(x) pour i = 0..N.
%
%
% ENTREE :
%   x : (scalaire) Point où évaluer les polynômes.
%   N : (entier) Degré maximal des polynômes à calculer.
%
% SORTIE :
%   W : (vecteur 1×(N+1)) contenant [W_0(x), W_1(x), ..., W_N(x)].
%
% =========================================================================

function W = cheb4_poly(x,N)
    W = zeros(1,N+1);
    W(1) = 1;         
    if N >= 1
        W(2) = 2*x + 1; 
    end
    for n = 2:N
        W(n+1) = 2*x*W(n) - W(n-1);
    end
end

% =========================================================================
% FONCTION : cheb4_poly_single
% ROLE : Calcule le polynôme de Tchebychev W_i(x)
%        pour un degré donné i et pour une valeur x.
% -------------------------------------------------------------------------
% DESCRIPTION :
%   Cette fonction implémente la définition récursive des polynômes de
%   Tchebychev
%
% ENTREE :
%   x : (scalaire ou vecteur) Point(s) où évaluer le polynôme.
%   i : (entier) Degré du polynôme W_i.
%
% SORTIE :
%   Wi : (scalaire ou vecteur) Valeur(s) du polynôme W_i(x).
%
% =========================================================================

function Wi = cheb4_poly_single(x,i)
    x = x(:)';
    if i == 0
        Wi = ones(size(x));
        return;
    elseif i == 1
        Wi = 2*x + 1;
        return;
    end

    % Initialisation
    Wm2 = ones(size(x));  
    Wm1 = 2*x + 1;            

    % Récurrence vectorielle
    for n = 2:i
        Wn = 2.*x.*Wm1 - Wm2;
        Wm2 = Wm1;
        Wm1 = Wn;
    end

    Wi = Wm1;
end

%%----Résolution de l'équation de Love ------
function [x_cheb , u_cheb]=solve_love_cheb4(N_cheb,K,a,b)                    
    x_cheb = linspace(a,b,N_cheb+1);    
    
    V = zeros(N_cheb+1,N_cheb+1);
    I = ones(N_cheb+1,1);  

    for j = 1:N_cheb+1
        xj = x_cheb(j);
        W_at_xj = cheb4_poly(xj,N_cheb);  
        
        for i = 0:N_cheb
            % W_i(xj)
            Wi_xj = W_at_xj(i+1);
            
            integrand_i = @(t) K(xj,t) .* cheb4_poly_single(t,i);
            
            I_i = integral(integrand_i,0,1,'AbsTol',1e-10,'RelTol',1e-8);
    
            V(j,i+1) = Wi_xj - I_i;
        end
    end
    C = V \ I;
    u_cheb = zeros(size(x_cheb));
    
    for k = 1:length(x_cheb)
        W_vals = cheb4_poly(x_cheb(k),N_cheb);   
        u_cheb(k) = W_vals * C;           
    end
end

[x_cheb,u_cheb]=solve_love_cheb4(100,K,a,b);

figure(13);
plot(x_cheb,u_cheb,'b-','LineWidth',2);
xlabel('x');
ylabel('u(x)');
title('Méthode de collocation de Tchebychev ');
grid on;

N_ref = 300;   
[x_ref, u_ref] = solve_love_cheb4(N_ref,K,a,b);

N_values = 2:2:40;    
errors = zeros(size(N_values));

%% Calcul de l'erreur pour chaque N
for k = 1:length(N_values)
    N = N_values(k);

    [xN, uN] = solve_love_cheb4(N,K,a,b);

    u_ref_interp = interp1(x_ref, u_ref, xN, 'linear');

    errors(k) = max(abs(uN - u_ref_interp));
end

figure(14);
semilogy(N_values, errors, 'b-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Degré N des polynômes de Tchebychev');
ylabel('Erreur ||u_N - u_{ref}||_\infty');
title('Convergence de la méthode de collocation');
grid on;




